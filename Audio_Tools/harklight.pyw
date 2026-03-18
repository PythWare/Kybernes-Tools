from __future__ import annotations

import os, json, queue, struct, threading, time, traceback
from dataclasses import dataclass, replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

"""Harklight is a KVS Audio tool, meant to be used with Aldnoah Engine"""

APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)

HEADER_STRUCT = struct.Struct("<4s7i")
VALID_MAGIC = {b"KOVS", b"KVS\x00"}
SIDECAR_MARKER = "kvs_tool_metadata"
SIDECAR_VERSION = 1
STREAM_CHUNK_SIZE = 1024 * 1024
QUEUE_SLICE_SECONDS = 0.012
LOG_FLUSH_INTERVAL_SECONDS = 0.12
PROGRESS_UPDATE_INTERVAL_SECONDS = 0.05

THEME = {
    "bg": "#090F1E",
    "bg_alt": "#0E1730",
    "panel": "#121C37",
    "panel_alt": "#0C1430",
    "edge": "#243765",
    "edge_bright": "#335395",
    "text": "#F3F6FF",
    "muted": "#94A7D8",
    "accent": "#FF8B3D",
    "accent_hover": "#FF9D58",
    "accent_2": "#48D3FF",
    "good": "#61D7A6",
    "warn": "#FFD166",
    "bad": "#FF647A",
    "button": "#172349",
    "button_hover": "#20315F",
    "button_active": "#FF8B3D",
    "button_text_active": "#0A1021",
    "log_bg": "#0A1125",
}


def create_round_rect(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    **kwargs,
):
    radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def xor_transform(blob: bytes) -> bytes:
    transformed = bytearray(blob)
    for index in range(min(0x100, len(transformed))):
        transformed[index] ^= index & 0xFF
    return bytes(transformed)


def looks_like_ogg(blob: bytes) -> bool:
    return len(blob) >= 4 and blob[:4] == b"OggS"


def read_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def xor_chunk_in_place(chunk: bytearray, start_offset: int):
    limit = min(len(chunk), max(0, 0x100 - start_offset))
    for index in range(limit):
        chunk[index] ^= (start_offset + index) & 0xFF


def copy_xor_stream(reader, writer, size: int):
    remaining = size
    offset = 0
    while remaining > 0:
        chunk = bytearray(reader.read(min(STREAM_CHUNK_SIZE, remaining)))
        if not chunk:
            raise ValueError("Unexpected end of file while streaming audio payload.")
        xor_chunk_in_place(chunk, offset)
        writer.write(chunk)
        offset += len(chunk)
        remaining -= len(chunk)


def parse_int(text: str, label: str) -> int:
    value = text.strip()
    if not value:
        raise ValueError(f"{label} cannot be blank.")
    try:
        return int(value, 0)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid integer.") from exc


@dataclass(frozen=True)
class HeaderValues:
    magic: bytes = b"KOVS"
    size: int = 0
    loop_start_samples: int = 0
    loop_end_samples: int = 0
    unknown1: int = 1
    unknown2: int = 0
    unknown3: int = 0
    unknown4: int = 0

    @classmethod
    def from_bytes(cls, blob: bytes) -> "HeaderValues":
        if len(blob) < HEADER_STRUCT.size:
            raise ValueError("File is too small to contain a KVS header.")
        magic, size, loop_start, loop_end, unknown1, unknown2, unknown3, unknown4 = HEADER_STRUCT.unpack(
            blob[: HEADER_STRUCT.size]
        )
        return cls(
            magic=magic,
            size=size,
            loop_start_samples=loop_start,
            loop_end_samples=loop_end,
            unknown1=unknown1,
            unknown2=unknown2,
            unknown3=unknown3,
            unknown4=unknown4,
        )

    @classmethod
    def from_metadata(cls, metadata: dict, size: int) -> "HeaderValues":
        magic_text = str(metadata.get("magic", "KOVS"))
        magic = magic_text.encode("ascii", "ignore")[:4].ljust(4, b"\x00")
        return cls(
            magic=magic or b"KOVS",
            size=size,
            loop_start_samples=int(metadata.get("loop_start_samples", 0)),
            loop_end_samples=int(metadata.get("loop_end_samples", 0)),
            unknown1=int(metadata.get("unknown1", 0)),
            unknown2=int(metadata.get("unknown2", 0)),
            unknown3=int(metadata.get("unknown3", 0)),
            unknown4=int(metadata.get("unknown4", 0)),
        )

    @property
    def magic_text(self) -> str:
        return self.magic.rstrip(b"\x00").decode("ascii", "replace")

    def with_size(self, size: int) -> "HeaderValues":
        return replace(self, size=size)

    def to_bytes(self) -> bytes:
        magic = self.magic[:4].ljust(4, b"\x00")
        return HEADER_STRUCT.pack(
            magic,
            self.size,
            self.loop_start_samples,
            self.loop_end_samples,
            self.unknown1,
            self.unknown2,
            self.unknown3,
            self.unknown4,
        )

    def to_metadata(self, source_path: Path, output_path: Path, trailer: bytes) -> dict:
        return {
            "format": SIDECAR_MARKER,
            "version": SIDECAR_VERSION,
            "source_file": source_path.name,
            "output_file": output_path.name,
            "container_extension": source_path.suffix.lower() or ".kvs",
            "magic": self.magic_text or "KOVS",
            "size": self.size,
            "loop_start_samples": self.loop_start_samples,
            "loop_end_samples": self.loop_end_samples,
            "unknown1": self.unknown1,
            "unknown2": self.unknown2,
            "unknown3": self.unknown3,
            "unknown4": self.unknown4,
            "trailer_hex": trailer.hex(),
        }


@dataclass(frozen=True)
class ParsedContainer:
    header: HeaderValues
    payload: bytes
    trailer: bytes


@dataclass(frozen=True)
class JobConfig:
    action: str
    mode: str
    input_path: Path
    output_path: Path | None
    recurse: bool
    write_sidecar: bool
    prefer_sidecar: bool
    default_loop_start: int
    default_loop_end: int


def build_default_header(size: int, loop_start: int, loop_end: int) -> HeaderValues:
    auto_unknown1 = 1 if loop_start == 0 and loop_end == 0 else 0
    return HeaderValues(
        magic=b"KOVS",
        size=size,
        loop_start_samples=loop_start,
        loop_end_samples=loop_end,
        unknown1=auto_unknown1,
        unknown2=0,
        unknown3=0,
        unknown4=0,
    )


def read_kvs_container(path: Path) -> ParsedContainer:
    data = path.read_bytes()
    header = HeaderValues.from_bytes(data)
    if header.magic not in VALID_MAGIC and header.magic_text != "KOVS":
        raise ValueError(f"{path.name} does not start with a supported KVS magic.")
    if header.size < 0:
        raise ValueError(f"{path.name} has a negative payload size in the header.")
    payload_end = HEADER_STRUCT.size + header.size
    if payload_end > len(data):
        raise ValueError(
            f"{path.name} declares {header.size} bytes of audio, but the file only has {len(data) - HEADER_STRUCT.size} bytes."
        )
    payload = data[HEADER_STRUCT.size:payload_end]
    trailer = data[payload_end:]
    return ParsedContainer(header=header, payload=payload, trailer=trailer)


def peek_kvs_container(path: Path) -> tuple[HeaderValues, int]:
    header = HeaderValues.from_bytes(read_prefix(path, HEADER_STRUCT.size))
    file_size = path.stat().st_size
    payload_end = HEADER_STRUCT.size + header.size
    if payload_end > file_size:
        raise ValueError(
            f"{path.name} declares {header.size} bytes of audio, but the file only has {file_size - HEADER_STRUCT.size} bytes."
        )
    return header, file_size - payload_end


def sidecar_candidates(source: Path) -> list[Path]:
    candidates = [
        Path(f"{source}.json"),
        source.with_suffix(".json"),
        source.with_name(f"{source.stem}.ogg.json"),
        source.with_name(f"{source.stem}.kvs.json"),
        source.with_name(f"{source.stem}.kovs.json"),
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def load_sidecar_for(source: Path) -> tuple[dict | None, Path | None]:
    for candidate in sidecar_candidates(source):
        if not candidate.exists():
            continue
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("format") == SIDECAR_MARKER:
            return metadata, candidate
    return None, None


def decrypt_file(source: Path, output_dir: Path | None, write_sidecar: bool) -> tuple[list[Path], str]:
    header, _trailer_len = peek_kvs_container(source)
    target = (output_dir / source.with_suffix(".ogg").name) if output_dir else source.with_suffix(".ogg")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("wb") as writer:
        reader.seek(HEADER_STRUCT.size)
        copy_xor_stream(reader, writer, header.size)
        trailer = reader.read()
    written = [target]
    notes = [
        f"{source.name} -> {target.name} ({human_size(header.size)})",
        f"loop_start={header.loop_start_samples}",
        f"loop_end={header.loop_end_samples}",
    ]
    if write_sidecar:
        sidecar_path = Path(f"{target}.json")
        sidecar_payload = header.to_metadata(source, target, trailer)
        sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")
        written.append(sidecar_path)
        notes.append(f"sidecar={sidecar_path.name}")
    if trailer:
        notes.append(f"trailer={len(trailer)}B")
    return written, " | ".join(notes)


def encrypt_file(
    source: Path,
    output_dir: Path | None,
    prefer_sidecar: bool,
    default_loop_start: int,
    default_loop_end: int,
) -> tuple[list[Path], str]:
    prefix = read_prefix(source, 4)
    if not looks_like_ogg(prefix):
        prefix_hex = prefix.hex(" ") if prefix else "empty"
        raise ValueError(
            f"{source.name} does not look like an OGG file. Pick the decrypted .ogg file, not a .kvs/.kovs container. "
            f"First bytes: {prefix_hex}"
        )
    ogg_size = source.stat().st_size
    metadata = None
    sidecar_path = None
    trailer = b""
    output_suffix = ".kvs"
    if prefer_sidecar:
        metadata, sidecar_path = load_sidecar_for(source)
    if metadata:
        header = HeaderValues.from_metadata(metadata, size=ogg_size)
        output_suffix = str(metadata.get("container_extension", ".kvs")) or ".kvs"
        trailer_hex = str(metadata.get("trailer_hex", ""))
        if trailer_hex:
            try:
                trailer = bytes.fromhex(trailer_hex)
            except ValueError as exc:
                raise ValueError(f"{sidecar_path.name} has invalid trailer_hex data.") from exc
    else:
        header = build_default_header(ogg_size, default_loop_start, default_loop_end)
    output_suffix = output_suffix if output_suffix.startswith(".") else f".{output_suffix}"
    target = (output_dir / f"{source.stem}{output_suffix}") if output_dir else source.with_suffix(output_suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as writer:
        writer.write(header.to_bytes())
        with source.open("rb") as reader:
            copy_xor_stream(reader, writer, ogg_size)
        if trailer:
            writer.write(trailer)
    note = [
        f"{source.name} -> {target.name} ({human_size(ogg_size)})",
        f"loop_start={header.loop_start_samples}",
        f"loop_end={header.loop_end_samples}",
        f"unknown1={header.unknown1}",
    ]
    if sidecar_path:
        note.append(f"metadata={sidecar_path.name}")
    if trailer:
        note.append(f"trailer={len(trailer)}B")
    return [target], " | ".join(note)


def collect_input_files(config: JobConfig) -> list[Path]:
    if config.mode == "single":
        return [config.input_path]

    patterns = ("*.kvs", "*.kovs") if config.action == "decrypt" else ("*.ogg",)
    iterator = config.input_path.rglob("*") if config.recurse else config.input_path.glob("*")
    input_root = config.input_path.resolve(strict=False)
    output_root = config.output_path.resolve(strict=False) if config.output_path else None
    exclude_output = bool(output_root and output_root != input_root and input_root in output_root.parents)
    found: list[Path] = []
    valid_suffixes = {pattern.replace("*", "").lower() for pattern in patterns}
    for path in iterator:
        if not path.is_file():
            continue
        if exclude_output and output_root in path.resolve(strict=False).parents:
            continue
        if path.suffix.lower() in valid_suffixes:
            found.append(path)
    return sorted(found)


class SegmentedToggle(tk.Frame):
    def __init__(self, master, variable: tk.StringVar, options, command=None):
        super().__init__(master, bg=THEME["bg"])
        self.variable = variable
        self.options = options
        self.command = command
        self.buttons: dict[str, tk.Button] = {}
        self.enabled = True
        shell = tk.Frame(self, bg=THEME["panel_alt"], highlightthickness=1, highlightbackground=THEME["edge"])
        shell.pack(fill="x")
        for value, label in options:
            button = tk.Button(
                shell,
                text=label,
                bd=0,
                relief="flat",
                cursor="hand2",
                font=("Bahnschrift SemiCondensed", 10, "bold"),
                padx=18,
                pady=10,
                command=lambda choice=value: self.select(choice),
            )
            button.pack(side="left", padx=4, pady=4, fill="x", expand=True)
            self.buttons[value] = button
        self.variable.trace_add("write", self.on_change)
        self.refresh()

    def on_change(self, *_args):
        self.refresh()
        if self.command:
            self.command(self.variable.get())

    def select(self, value: str):
        if self.enabled:
            self.variable.set(value)

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.refresh()

    def refresh(self):
        current = self.variable.get()
        for value, button in self.buttons.items():
            active = value == current
            if not self.enabled:
                button.configure(
                    state="disabled",
                    bg=THEME["panel"],
                    fg=THEME["muted"],
                    activebackground=THEME["panel"],
                    activeforeground=THEME["muted"],
                )
            elif active:
                button.configure(
                    state="normal",
                    bg=THEME["button_active"],
                    fg=THEME["button_text_active"],
                    activebackground=THEME["accent_hover"],
                    activeforeground=THEME["button_text_active"],
                )
            else:
                button.configure(
                    state="normal",
                    bg=THEME["button"],
                    fg=THEME["text"],
                    activebackground=THEME["button_hover"],
                    activeforeground=THEME["text"],
                )


class ProgressStrip(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, height=34, bg=THEME["panel"], highlightthickness=0)
        self.progress = 0.0
        self.label = "Idle"
        self.bind("<Configure>", self.redraw)

    def set(self, progress: float, label: str):
        self.progress = max(0.0, min(1.0, progress))
        self.label = label
        self.redraw()

    def redraw(self, _event=None):
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = max(self.winfo_height(), 10)
        create_round_rect(self, 0, 0, width, height, 14, fill=THEME["bg_alt"], outline="")
        create_round_rect(self, 2, 2, width - 2, height - 2, 12, fill=THEME["log_bg"], outline="")
        fill_width = max(8, (width - 4) * self.progress) if self.progress > 0 else 0
        if fill_width:
            create_round_rect(self, 2, 2, 2 + fill_width, height - 2, 12, fill=THEME["accent"], outline="")
            self.create_line(18, height / 2, max(18, fill_width - 18), height / 2, fill=THEME["accent_2"], width=2)
        self.create_text(
            16,
            height / 2,
            anchor="w",
            text=self.label,
            font=("Segoe UI Semibold", 10),
            fill=THEME["text"],
        )


class KVSToolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Harklight, KVS Audio Tool")
        self.root.geometry("1120x1030")
        self.root.minsize(980, 700)
        self.root.configure(bg=THEME["bg"])

        self.action_var = tk.StringVar(value="decrypt")
        self.mode_var = tk.StringVar(value="single")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.recurse_var = tk.BooleanVar(value=False)
        self.write_sidecar_var = tk.BooleanVar(value=True)
        self.prefer_sidecar_var = tk.BooleanVar(value=True)
        self.loop_start_var = tk.StringVar(value="0")
        self.loop_end_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready")
        self.input_hint_var = tk.StringVar(value="Pick a KVS/KOVS file to decrypt or an OGG file to package.")

        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.hero_phase = 0
        self.last_progress_update = 0.0
        self.last_log_flush = 0.0
        self.pending_log_lines: list[tuple[str, str]] = []

        self.busy_widgets: list[tk.Widget] = []
        self.segment_controls: list[SegmentedToggle] = []

        self.build_ui()
        self.schedule_queue_poll()
        self._animate_hero()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        self.backdrop = tk.Canvas(self.root, bg=THEME["bg"], highlightthickness=0)
        self.backdrop.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.backdrop.bind("<Configure>", self.draw_backdrop)

        self.shell = tk.Frame(self.root, bg=THEME["bg"])
        self.shell.pack(fill="both", expand=True, padx=24, pady=20)

        self.hero = tk.Canvas(self.shell, height=170, bg=THEME["bg"], highlightthickness=0)
        self.hero.pack(fill="x")
        self.hero.bind("<Configure>", self.draw_hero)

        toggles = tk.Frame(self.shell, bg=THEME["bg"])
        toggles.pack(fill="x", pady=(14, 18))
        toggles.grid_columnconfigure(0, weight=1)
        toggles.grid_columnconfigure(1, weight=1)

        action_group = self.make_card(toggles, "Operation", "Choose the direction for the conversion.")
        action_group["card"].grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.action_toggle = SegmentedToggle(
            action_group["body"],
            self.action_var,
            [("decrypt", "Decrypt to OGG"), ("encrypt", "Create KVS")],
            command=lambda _value: self.on_mode_changed(),
        )
        self.action_toggle.pack(fill="x")
        self.segment_controls.append(self.action_toggle)

        mode_group = self.make_card(toggles, "Scope", "Run one file at a time or sweep a whole folder.")
        mode_group["card"].grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.mode_toggle = SegmentedToggle(
            mode_group["body"],
            self.mode_var,
            [("single", "Single file"), ("folder", "Folder batch")],
            command=lambda _value: self.on_mode_changed(),
        )
        self.mode_toggle.pack(fill="x")
        self.segment_controls.append(self.mode_toggle)

        content = tk.Frame(self.shell, bg=THEME["bg"])
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(2, weight=1)

        io_card = self.make_card(content, "Source/Output", "Pick your input and where the converted files should go.")
        io_card["card"].grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 12))
        self.build_io_section(io_card["body"])

        options_card = self.make_card(
            content,
            "Packaging Rules",
            "Metadata sidecars keep loop points and unknown header fields intact for round-trips.",
        )
        options_card["card"].grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 12))
        self.build_options_section(options_card["body"])

        progress_card = self.make_card(content, "Run Status", "Large files process in a background thread so the window stays responsive.")
        progress_card["card"].grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.build_progress_section(progress_card["body"])

        log_card = self.make_card(content, "Conversion Log", "Every file result lands here, including metadata reuse and header previews.")
        log_card["card"].grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.build_log_section(log_card["body"])

        self.on_mode_changed()
        self.append_log("Loose KVS files in this workspace use a 32-byte header and XOR only the first 0x100 payload bytes.", "accent")

    def make_card(self, parent, title: str, subtitle: str):
        card = tk.Frame(parent, bg=THEME["panel"], highlightthickness=1, highlightbackground=THEME["edge"])
        header = tk.Frame(card, bg=THEME["panel"])
        header.pack(fill="x", padx=18, pady=(16, 6))
        tk.Label(
            header,
            text=title,
            bg=THEME["panel"],
            fg=THEME["text"],
            font=("Bahnschrift SemiCondensed", 15, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=subtitle,
            bg=THEME["panel"],
            fg=THEME["muted"],
            font=("Segoe UI", 9),
            wraplength=460,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))
        body = tk.Frame(card, bg=THEME["panel"])
        body.pack(fill="both", expand=True, padx=18, pady=(6, 18))
        return {"card": card, "body": body}

    def build_io_section(self, parent):
        parent.grid_columnconfigure(0, weight=1)

        self.build_path_row(
            parent,
            row=0,
            label="Input",
            variable=self.input_var,
            button_text="Browse",
            command=self.choose_input,
        )
        tk.Label(
            parent,
            textvariable=self.input_hint_var,
            bg=THEME["panel"],
            fg=THEME["muted"],
            font=("Segoe UI", 9),
            justify="left",
            wraplength=540,
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        self.recurse_check = tk.Checkbutton(
            parent,
            text="Include subfolders when batch scanning",
            variable=self.recurse_var,
            command=self.refresh_input_hint,
            bg=THEME["panel"],
            fg=THEME["text"],
            selectcolor=THEME["panel_alt"],
            activebackground=THEME["panel"],
            activeforeground=THEME["text"],
            font=("Segoe UI", 9),
        )
        self.recurse_check.grid(row=2, column=0, sticky="w", pady=(0, 14))
        self.busy_widgets.append(self.recurse_check)

        self.build_path_row(
            parent,
            row=3,
            label="Output folder (optional)",
            variable=self.output_var,
            button_text="Pick folder",
            command=self.choose_output_folder,
        )
        self.output_hint_label = tk.Label(
            parent,
            text="Leave this blank to write beside the source file or inside the source folder.",
            bg=THEME["panel"],
            fg=THEME["muted"],
            font=("Segoe UI", 9),
            justify="left",
            wraplength=540,
        )
        self.output_hint_label.grid(row=4, column=0, sticky="w", pady=(4, 18))

        actions = tk.Frame(parent, bg=THEME["panel"])
        actions.grid(row=5, column=0, sticky="ew")
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)

        self.run_button = tk.Button(
            actions,
            text="Start Conversion",
            command=self.start_job,
            bd=0,
            relief="flat",
            bg=THEME["accent"],
            fg=THEME["button_text_active"],
            activebackground=THEME["accent_hover"],
            activeforeground=THEME["button_text_active"],
            font=("Bahnschrift SemiCondensed", 13, "bold"),
            pady=12,
            cursor="hand2",
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.cancel_button = tk.Button(
            actions,
            text="Cancel",
            command=self.cancel_job,
            bd=0,
            relief="flat",
            bg=THEME["button"],
            fg=THEME["text"],
            activebackground=THEME["button_hover"],
            activeforeground=THEME["text"],
            font=("Bahnschrift SemiCondensed", 13, "bold"),
            pady=12,
            cursor="hand2",
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.busy_widgets.extend([self.run_button, self.cancel_button])

    def build_path_row(self, parent, row: int, label: str, variable: tk.StringVar, button_text: str, command):
        row_frame = tk.Frame(parent, bg=THEME["panel"])
        row_frame.grid(row=row, column=0, sticky="ew", pady=(0, 0))
        row_frame.grid_columnconfigure(0, weight=1)

        tk.Label(
            row_frame,
            text=label,
            bg=THEME["panel"],
            fg=THEME["text"],
            font=("Segoe UI Semibold", 10),
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        input_shell = tk.Frame(
            row_frame,
            bg=THEME["panel_alt"],
            highlightthickness=1,
            highlightbackground=THEME["edge"],
        )
        input_shell.grid(row=1, column=0, sticky="ew")
        input_shell.grid_columnconfigure(0, weight=1)

        entry = tk.Entry(
            input_shell,
            textvariable=variable,
            bd=0,
            relief="flat",
            bg=THEME["panel_alt"],
            fg=THEME["text"],
            insertbackground=THEME["accent_2"],
            font=("Consolas", 10),
        )
        entry.grid(row=0, column=0, sticky="ew", padx=12, pady=10)
        browse = tk.Button(
            input_shell,
            text=button_text,
            command=command,
            bd=0,
            relief="flat",
            bg=THEME["button"],
            fg=THEME["text"],
            activebackground=THEME["button_hover"],
            activeforeground=THEME["text"],
            font=("Segoe UI Semibold", 9),
            padx=14,
            pady=8,
            cursor="hand2",
        )
        browse.grid(row=0, column=1, sticky="e", padx=8, pady=6)
        self.busy_widgets.extend([entry, browse])
        if label.startswith("Input"):
            self.input_entry = entry
            self.input_browse_button = browse
        else:
            self.output_entry = entry
            self.output_browse_button = browse

    def build_options_section(self, parent):
        self.decrypt_options = tk.Frame(parent, bg=THEME["panel"])
        self.encrypt_options = tk.Frame(parent, bg=THEME["panel"])

        self.decrypt_sidecar_check = tk.Checkbutton(
            self.decrypt_options,
            text="Write a matching .json sidecar when decrypting",
            variable=self.write_sidecar_var,
            bg=THEME["panel"],
            fg=THEME["text"],
            selectcolor=THEME["panel_alt"],
            activebackground=THEME["panel"],
            activeforeground=THEME["text"],
            font=("Segoe UI", 10),
        )
        self.decrypt_sidecar_check.pack(anchor="w")
        tk.Label(
            self.decrypt_options,
            text="The sidecar keeps loop points, unknown header fields, original extension, and any trailing bytes so edited OGG files can be packed back cleanly.",
            bg=THEME["panel"],
            fg=THEME["muted"],
            font=("Segoe UI", 9),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))
        self.busy_widgets.append(self.decrypt_sidecar_check)

        self.encrypt_sidecar_check = tk.Checkbutton(
            self.encrypt_options,
            text="Prefer matching sidecar metadata when packaging",
            variable=self.prefer_sidecar_var,
            bg=THEME["panel"],
            fg=THEME["text"],
            selectcolor=THEME["panel_alt"],
            activebackground=THEME["panel"],
            activeforeground=THEME["text"],
            font=("Segoe UI", 10),
        )
        self.encrypt_sidecar_check.pack(anchor="w")
        tk.Label(
            self.encrypt_options,
            text="If a sidecar is found next to the OGG, it wins. Otherwise the tool builds a clean KOVS header from the loop values below and defaults the unknown fields automatically.",
            bg=THEME["panel"],
            fg=THEME["muted"],
            font=("Segoe UI", 9),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(8, 14))
        self.busy_widgets.append(self.encrypt_sidecar_check)

        field_grid = tk.Frame(self.encrypt_options, bg=THEME["panel"])
        field_grid.pack(fill="x")
        field_grid.grid_columnconfigure(0, weight=1)
        field_grid.grid_columnconfigure(1, weight=1)

        self.build_number_field(field_grid, 0, 0, "Loop start samples", self.loop_start_var)
        self.build_number_field(field_grid, 0, 1, "Loop end samples", self.loop_end_var)

        tk.Label(
            self.encrypt_options,
            text="Auto header rule: unknown1 becomes 1 for non-looping files and 0 when either loop point is set. Unknown2-4 default to 0 unless a sidecar overrides them.",
            bg=THEME["panel"],
            fg=THEME["muted"],
            font=("Segoe UI", 9),
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(14, 0))

        self.decrypt_options.pack(fill="x")

    def build_number_field(self, parent, row: int, column: int, label: str, variable: tk.StringVar):
        shell = tk.Frame(parent, bg=THEME["panel"])
        shell.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else 8), pady=(0, 0))
        tk.Label(
            shell,
            text=label,
            bg=THEME["panel"],
            fg=THEME["text"],
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", pady=(0, 6))
        entry_shell = tk.Frame(shell, bg=THEME["panel_alt"], highlightthickness=1, highlightbackground=THEME["edge"])
        entry_shell.pack(fill="x")
        entry = tk.Entry(
            entry_shell,
            textvariable=variable,
            bd=0,
            relief="flat",
            bg=THEME["panel_alt"],
            fg=THEME["text"],
            insertbackground=THEME["accent_2"],
            font=("Consolas", 10),
        )
        entry.pack(fill="x", padx=10, pady=10)
        self.busy_widgets.append(entry)

    def build_progress_section(self, parent):
        status_row = tk.Frame(parent, bg=THEME["panel"])
        status_row.pack(fill="x")
        self.status_badge = tk.Label(
            status_row,
            textvariable=self.status_var,
            bg=THEME["button"],
            fg=THEME["text"],
            font=("Segoe UI Semibold", 10),
            padx=14,
            pady=8,
        )
        self.status_badge.pack(side="left")

        self.progress = ProgressStrip(parent)
        self.progress.pack(fill="x", pady=(14, 0))
        self.progress.set(0.0, "Idle")

    def build_log_section(self, parent):
        shell = tk.Frame(parent, bg=THEME["log_bg"], highlightthickness=1, highlightbackground=THEME["edge"])
        shell.pack(fill="both", expand=True)
        self.log = tk.Text(
            shell,
            bg=THEME["log_bg"],
            fg=THEME["text"],
            relief="flat",
            bd=0,
            wrap="word",
            font=("Consolas", 10),
            insertbackground=THEME["accent_2"],
            padx=12,
            pady=12,
        )
        scrollbar = tk.Scrollbar(shell, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.log.tag_configure("accent", foreground=THEME["accent_2"])
        self.log.tag_configure("good", foreground=THEME["good"])
        self.log.tag_configure("bad", foreground=THEME["bad"])
        self.log.tag_configure("warn", foreground=THEME["warn"])
        self.log.tag_configure("muted", foreground=THEME["muted"])

    def draw_backdrop(self, _event=None):
        self.backdrop.delete("all")
        width = self.backdrop.winfo_width()
        height = self.backdrop.winfo_height()
        self.backdrop.create_rectangle(0, 0, width, height, fill=THEME["bg"], outline="")
        self.backdrop.create_oval(-220, -140, 260, 280, fill=THEME["bg_alt"], outline="")
        self.backdrop.create_oval(width - 340, -120, width + 80, 260, fill="#10214A", outline="")
        self.backdrop.create_oval(width - 420, height - 300, width + 80, height + 160, fill="#0D1836", outline="")
        self.backdrop.create_line(0, height * 0.24, width, height * 0.06, fill=THEME["edge"], width=2)
        self.backdrop.create_line(0, height * 0.84, width, height * 0.56, fill=THEME["edge"], width=1)

    def draw_hero(self, _event=None):
        self.hero.delete("all")
        width = max(self.hero.winfo_width(), 300)
        height = max(self.hero.winfo_height(), 150)
        create_round_rect(self.hero, 0, 0, width, height, 24, fill=THEME["panel"], outline="")
        create_round_rect(self.hero, 2, 2, width - 2, height - 2, 22, fill=THEME["panel"], outline=THEME["edge"], width=1)
        self.hero.create_rectangle(0, 0, width, 10, fill=THEME["accent"], outline="")
        self.hero.create_oval(width - 190, 22, width - 48, 164, outline=THEME["accent_2"], width=2)
        self.hero.create_oval(width - 164, 48, width - 74, 138, outline=THEME["edge_bright"], width=2)
        for offset in range(5):
            x1 = width - 330 + ((self.hero_phase * 16) + offset * 42) % 280
            self.hero.create_line(x1, 18, x1 - 76, 150, fill=THEME["edge"], width=1)
        self.hero.create_text(
            34,
            44,
            anchor="w",
            text="KVS/Audio Toolkit",
            fill=THEME["text"],
            font=("Bahnschrift SemiCondensed", 26, "bold"),
        )
        self.hero.create_text(
            36,
            84,
            anchor="w",
            text="Decrypt loose KVS/KOVS audio to OGG, then pack edited OGG files back into the same container format.",
            fill=THEME["muted"],
            font=("Segoe UI", 11),
            width=560,
        )

    def _animate_hero(self):
        if self.is_busy:
            self.root.after(180, self._animate_hero)
            return
        self.hero_phase = (self.hero_phase + 1) % 40
        self.draw_hero()
        self.root.after(100, self._animate_hero)

    def append_log(self, message: str, tag: str = "muted"):
        self.log.insert("end", message + "\n", tag)
        self.log.see("end")

    def flush_pending_logs(self, force: bool = False):
        if not self.pending_log_lines:
            return
        now = time.perf_counter()
        if not force and now - self.last_log_flush < LOG_FLUSH_INTERVAL_SECONDS:
            return
        self.last_log_flush = now
        for message, tag in self.pending_log_lines:
            self.log.insert("end", message + "\n", tag)
        self.pending_log_lines.clear()
        self.log.see("end")

    def choose_input(self):
        initial_dir = self.dialog_initial_dir(self.input_var.get())
        if self.mode_var.get() == "single":
            if self.action_var.get() == "decrypt":
                path = filedialog.askopenfilename(
                    title="Pick a KVS/KOVS file",
                    filetypes=[("KVS audio", "*.kvs *.kovs"), ("All files", "*.*")],
                    initialdir=initial_dir,
                )
            else:
                path = filedialog.askopenfilename(
                    title="Pick an OGG file",
                    filetypes=[("OGG audio", "*.ogg"), ("All files", "*.*")],
                    initialdir=initial_dir,
                )
        else:
            path = filedialog.askdirectory(title="Pick a folder to batch process", initialdir=initial_dir)
        if path:
            self.input_var.set(path)
            self.refresh_input_hint()

    def choose_output_folder(self):
        seed = self.output_var.get() or self.input_var.get()
        path = filedialog.askdirectory(title="Pick an output folder", initialdir=self.dialog_initial_dir(seed))
        if path:
            self.output_var.set(path)

    def dialog_initial_dir(self, seed: str | None = None) -> str:
        if seed:
            path = Path(seed)
            if path.exists():
                return str(path if path.is_dir() else path.parent)
        return str(APP_DIR)

    def refresh_input_hint(self):
        input_text = self.input_var.get().strip()
        if not input_text:
            self.input_hint_var.set("Pick a file or folder to preview what the tool will process.")
            return

        path = Path(input_text)
        if not path.exists():
            self.input_hint_var.set("The selected path does not exist yet.")
            return

        try:
            if self.mode_var.get() == "single":
                if self.action_var.get() == "decrypt":
                    header, trailer_len = peek_kvs_container(path)
                    self.input_hint_var.set(
                        f"{path.name}: payload={human_size(header.size)}, loop_start={header.loop_start_samples}, "
                        f"loop_end={header.loop_end_samples}, unknown1={header.unknown1}, trailer={trailer_len}B"
                    )
                else:
                    if not looks_like_ogg(read_prefix(path, 4)):
                        self.input_hint_var.set(
                            f"{path.name}: this does not look like an OGG file. Pick the decrypted .ogg, not the .kvs container."
                        )
                    else:
                        metadata, sidecar_path = load_sidecar_for(path)
                        if metadata and sidecar_path:
                            self.input_hint_var.set(
                                f"{path.name}: {human_size(path.stat().st_size)} OGG, sidecar found ({sidecar_path.name}), "
                                f"loop_start={metadata.get('loop_start_samples', 0)}, loop_end={metadata.get('loop_end_samples', 0)}"
                            )
                        else:
                            self.input_hint_var.set(
                                f"{path.name}: {human_size(path.stat().st_size)} OGG, no sidecar found. Manual loop values will be used."
                            )
            else:
                recurse_note = "including subfolders" if self.recurse_var.get() else "top folder only"
                label = "KVS/KOVS containers" if self.action_var.get() == "decrypt" else "OGG files"
                self.input_hint_var.set(
                    f"{path.name}: ready for batch processing. Matching {label} will be discovered in the worker thread ({recurse_note}) when you press Start."
                )
        except Exception as exc:
            self.input_hint_var.set(f"Preview failed: {exc}")

    def on_mode_changed(self):
        folder_mode = self.mode_var.get() == "folder"
        self.recurse_check.configure(state="normal" if folder_mode and not self.is_busy else "disabled")
        if self.action_var.get() == "decrypt":
            self.encrypt_options.pack_forget()
            self.decrypt_options.pack(fill="x")
        else:
            self.decrypt_options.pack_forget()
            self.encrypt_options.pack(fill="x")
        if folder_mode:
            self.output_hint_label.configure(text="Leave this blank to write the converted files inside the selected source folder.")
        else:
            self.output_hint_label.configure(text="Leave this blank to write beside the selected source file.")
        self.refresh_input_hint()

    @property
    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def set_busy(self, busy: bool):
        self.run_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        widget_state = "disabled" if busy else "normal"
        for widget in self.busy_widgets:
            if widget in {self.run_button, self.cancel_button}:
                continue
            try:
                widget.configure(state=widget_state)
            except tk.TclError:
                pass
        for control in self.segment_controls:
            control.set_enabled(not busy)
        self.recurse_check.configure(state="normal" if (self.mode_var.get() == "folder" and not busy) else "disabled")
        self.status_badge.configure(bg=THEME["warn"] if busy else THEME["button"], fg=THEME["bg"] if busy else THEME["text"])

    def build_config(self) -> JobConfig:
        input_text = self.input_var.get().strip()
        if not input_text:
            raise ValueError("Pick an input file or folder first.")
        input_path = Path(input_text)
        if not input_path.exists():
            raise ValueError("The chosen input path does not exist.")
        output_text = self.output_var.get().strip()
        output_path = Path(output_text) if output_text else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        if self.mode_var.get() == "single" and input_path.is_dir():
            raise ValueError("Single-file mode needs a file, not a folder.")
        if self.mode_var.get() == "folder" and not input_path.is_dir():
            raise ValueError("Folder batch mode needs a folder, not a file.")
        if self.mode_var.get() == "single":
            if self.action_var.get() == "decrypt" and input_path.suffix.lower() not in {".kvs", ".kovs"}:
                raise ValueError("Decrypt mode expects a .kvs or .kovs file.")
            if self.action_var.get() == "encrypt":
                if input_path.suffix.lower() != ".ogg":
                    raise ValueError("Encrypt mode expects a decrypted .ogg file.")
                if not looks_like_ogg(read_prefix(input_path, 4)):
                    raise ValueError("The selected file does not start with OggS. Pick the decrypted .ogg file, not the .kvs container.")
        if self.action_var.get() == "encrypt":
            default_loop_start = parse_int(self.loop_start_var.get(), "Loop start samples")
            default_loop_end = parse_int(self.loop_end_var.get(), "Loop end samples")
        else:
            default_loop_start = 0
            default_loop_end = 0
        return JobConfig(
            action=self.action_var.get(),
            mode=self.mode_var.get(),
            input_path=input_path,
            output_path=output_path,
            recurse=self.recurse_var.get(),
            write_sidecar=self.write_sidecar_var.get(),
            prefer_sidecar=self.prefer_sidecar_var.get(),
            default_loop_start=default_loop_start,
            default_loop_end=default_loop_end,
        )

    def start_job(self):
        if self.is_busy:
            return
        try:
            config = self.build_config()
        except Exception as exc:
            messagebox.showerror("Cannot start conversion", str(exc), parent=self.root)
            return

        self.cancel_event = threading.Event()
        self.pending_log_lines.clear()
        self.last_progress_update = 0.0
        self.last_log_flush = 0.0
        self.set_busy(True)
        self.status_var.set("Running")
        if config.mode == "folder":
            self.progress.set(0.0, "Scanning folder in worker thread")
            self.append_log(f"Starting {config.action} batch job. File discovery will happen in the worker thread.", "accent")
        else:
            self.progress.set(0.0, f"Queued {config.input_path.name}")
            self.append_log(f"Starting {config.action} job for {config.input_path.name}.", "accent")
        self.worker = threading.Thread(target=self.run_job, args=(config,), daemon=True)
        self.worker.start()

    def cancel_job(self):
        if self.is_busy:
            self.cancel_event.set()
            self.status_var.set("Cancelling")
            self.append_log("Cancellation requested. The current file will finish safely before the worker stops.", "warn")

    def run_job(self, config: JobConfig):
        successes = 0
        failures = 0
        try:
            files = collect_input_files(config)
            if not files:
                raise ValueError("No matching files were found for the selected action.")
            total = len(files)
            self.events.put(("scan_complete", total, config.action))
            for index, source in enumerate(files, start=1):
                if self.cancel_event.is_set():
                    self.events.put(("cancelled", successes, failures, total))
                    return
                try:
                    if config.action == "decrypt":
                        _outputs, note = decrypt_file(source, config.output_path, config.write_sidecar)
                    else:
                        _outputs, note = encrypt_file(
                            source,
                            config.output_path,
                            config.prefer_sidecar,
                            config.default_loop_start,
                            config.default_loop_end,
                        )
                    successes += 1
                    self.events.put(("log", note, "good"))
                    self.events.put(("progress", index, total, f"{index}/{total} - {source.name}"))
                except Exception as exc:
                    failures += 1
                    self.events.put(("log", f"{source.name} failed: {exc}", "bad"))
                    self.events.put(("progress", index, total, f"{index}/{total} - {source.name}"))
            self.events.put(("done", successes, failures, total))
        except ValueError as exc:
            self.events.put(("job_error", str(exc)))
        except Exception:
            self.events.put(("fatal", traceback.format_exc()))

    def schedule_queue_poll(self):
        self.poll_queue()
        self.root.after(90, self.schedule_queue_poll)

    def poll_queue(self):
        deadline = time.perf_counter() + QUEUE_SLICE_SECONDS
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "log":
                _, message, tag = event
                self.pending_log_lines.append((message, tag))
                self.flush_pending_logs()
            elif kind == "scan_complete":
                _, total, action = event
                noun = "container" if action == "decrypt" else "OGG"
                plural = "" if total == 1 else "s"
                self.progress.set(0.0, f"Found {total} {noun}{plural}. Processing")
                self.pending_log_lines.append((f"Worker found {total} matching file(s).", "muted"))
                self.flush_pending_logs(force=True)
            elif kind == "progress":
                _, index, total, label = event
                now = time.perf_counter()
                if now - self.last_progress_update >= PROGRESS_UPDATE_INTERVAL_SECONDS or index == total:
                    progress = 0 if total == 0 else index / total
                    self.progress.set(progress, label)
                    self.last_progress_update = now
            elif kind == "job_error":
                _, message = event
                self.flush_pending_logs(force=True)
                self.set_busy(False)
                self.status_var.set("Stopped")
                self.progress.set(0.0, message)
                self.append_log(message, "bad")
                messagebox.showerror("Conversion could not start", message, parent=self.root)
                self.worker = None
            elif kind == "done":
                _, successes, failures, total = event
                self.flush_pending_logs(force=True)
                self.set_busy(False)
                self.status_var.set("Finished")
                self.progress.set(1.0, f"Completed {successes}/{total} file(s)")
                summary = f"Finished. Successes: {successes}, Failures: {failures}."
                self.append_log(summary, "accent")
                if failures:
                    messagebox.showwarning("Conversion finished with issues", summary, parent=self.root)
                self.worker = None
            elif kind == "cancelled":
                _, successes, failures, total = event
                self.flush_pending_logs(force=True)
                self.set_busy(False)
                self.status_var.set("Cancelled")
                self.progress.set(0.0, f"Stopped after {successes + failures}/{total} file(s)")
                self.append_log(f"Job cancelled. Completed {successes} file(s), {failures} failed before stop.", "warn")
                self.worker = None
            elif kind == "fatal":
                _, details = event
                self.flush_pending_logs(force=True)
                self.set_busy(False)
                self.status_var.set("Error")
                self.progress.set(0.0, "Stopped by an unexpected error")
                self.append_log("Unexpected worker error:\n" + details, "bad")
                messagebox.showerror("Unexpected error", details, parent=self.root)
                self.worker = None
            if time.perf_counter() >= deadline:
                break
        self.flush_pending_logs()

    def on_close(self):
        if self.is_busy:
            should_close = messagebox.askyesno(
                "Conversion still running",
                "Cancel the current job and close the window?",
                parent=self.root,
            )
            if not should_close:
                return
            self.cancel_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    KVSToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
