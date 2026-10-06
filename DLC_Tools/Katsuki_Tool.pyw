import ctypes, json, os, queue, re, struct, sys, threading, time, traceback
from pathlib import Path

APP_NAME = "Katsuki's Software"
SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
CRASH_LOG_NAME = "katsuki_tool_crash.log"

for stream_name in ("stdout", "stderr"):
    if getattr(sys, stream_name) is None:
        setattr(sys, stream_name, open(os.devnull, "w", encoding="utf-8"))

def write_crash_log(text):
    folders = [SCRIPT_DIR]
    temp_dir = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp_dir:
        folders.append(Path(temp_dir))
    for folder in folders:
        path = folder / CRASH_LOG_NAME
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Python {sys.version.split()[0]} at {sys.executable}\n")
                handle.write(text.rstrip() + "\n\n")
            return path
        except OSError:
            continue
    return None

def show_fatal(message):
    try:
        ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
    except Exception:
        sys.stderr.write(message + "\n")

def fatal_exit(message, detail=""):
    log_path = write_crash_log(f"{message}\n{detail}")
    if log_path:
        message = f"{message}\n\nDetails were written to:\n{log_path}"
    show_fatal(message)
    sys.exit(1)

if sys.version_info < (3, 8):
    fatal_exit(
        f"{APP_NAME} needs Python 3.8 or newer but it was opened with Python {sys.version.split()[0]}.\n\n"
        "Install a current Python from python.org, then double click the tool again."
    )

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
    from tkinter import font as tkfont
except ImportError:
    fatal_exit(
        f"This Python install has no working Tkinter, so {APP_NAME} cant open its window.\n\n"
        "Rerun the python.org installer, choose Modify, and make sure \"tcl/tk and IDLE\" is checked.",
        traceback.format_exc(),
    )

BG = "#12100F"
BG_ALT = "#191411"
PANEL = "#231914"
PANEL_ALT = "#2F221B"
PANEL_SOFT = "#3A2A20"
TEXT = "#F8F1E6"
TEXT_MUTED = "#C7B69C"
TEXT_DARK = "#1B140F"
ACCENT = "#FF6A13"
ACCENT_BRIGHT = "#FFA12E"
ACCENT_DEEP = "#D94715"
GREEN = "#5F7934"
GREEN_BRIGHT = "#89A247"
WARNING = "#F08A22"
ERROR_TEXT = "#FF5A40"
METAL = "#9A9BA2"
BORDER = "#6F4A25"

LEVEL_COLORS = {
    "info": TEXT,
    "muted": TEXT_MUTED,
    "good": GREEN_BRIGHT,
    "warn": WARNING,
    "error": ERROR_TEXT,
}

DLC_HEADER = struct.Struct("<4I")
SLOT_COUNT = 32
OFFSET_TABLE = DLC_HEADER.size
SIZE_TABLE = OFFSET_TABLE + SLOT_COUNT * 4
DATA_START = SIZE_TABLE + SLOT_COUNT * 4
DLC_ALIGN = 16
DEFAULT_HEADER = 100

MANIFEST_NAME = "katsuki_dlc.json"
INDEX_PATTERN = re.compile(r"^(\d+)")

SUB_MANIFEST = "katsuki_sub.json"
SUB_STRINGS = "strings.txt"
SUB_ALIGNS = (2048, 256, 128, 64, 32, 16, 8, 4)
SUB_MAX_COUNT = 4096
SUB_MAX_DEPTH = 6
STRING_ESCAPES = {"\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
STRING_UNESCAPES = {"\\": "\\", "n": "\n", "r": "\r", "t": "\t"}

EXT4 = {
    b"GT1G": ".g1t",
    b"_M1G": ".g1m",
    b"_S1G": ".g1s",
    b"_A1G": ".g1a",
    b"_A2G": ".g2a",
    b"_E1G": ".g1e",
    b"ME1G": ".g1em",
    b"XF1G": ".g1fx",
    b"OC1G": ".g1c",
    b"_L1G": ".g1l",
    b"_N1G": ".g1n",
    b"_H1G": ".g1h",
    b"SV1G": ".g1vs",
    b"_MHK": ".khm",
    b"KFTK": ".ktf",
    b"KTSR": ".ktsl2stbin",
    b"KTSC": ".ktsl2asbin",
    b"KTSS": ".ktss",
    b"_HBW": ".wbh",
    b"_DBW": ".wbd",
    b"_OLS": ".sebin",
    b"_COK": ".koc",
    b"LHSK": ".kshl",
    b"_SPK": ".postfx",
    b"OggS": ".ogg",
    b"DDS ": ".dds",
}

class DlcError(Exception):
    pass

def align_up(value, alignment=DLC_ALIGN):
    return (value + alignment - 1) & ~(alignment - 1)

def detect_ext(data):
    head = data[:4]
    if head in EXT4:
        return EXT4[head]
    if head == b"RIFF":
        return ".wav" if b"WAVE" in data[:16] else ".riff"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return ".bin"

def read_dlc(path):
    data = Path(path).read_bytes()
    if len(data) < DATA_START:
        raise DlcError(f"only {len(data)} bytes, too damn small for a DLC table")

    header, count, reserved_a, reserved_b = DLC_HEADER.unpack_from(data, 0)
    if not 0 < count <= SLOT_COUNT:
        raise DlcError(f"entry count {count} is outside {SLOT_COUNT}")

    offsets = struct.unpack_from(f"<{SLOT_COUNT}I", data, OFFSET_TABLE)
    sizes = struct.unpack_from(f"<{SLOT_COUNT}I", data, SIZE_TABLE)
    if any(offsets[count:]) or any(sizes[count:]):
        raise DlcError("unused table slots arent empty")

    entries = []
    for index in range(count):
        offset, size = offsets[index], sizes[index]
        if offset < DATA_START or offset + size > len(data):
            raise DlcError(f"entry {index} points outside the file")
        entries.append(data[offset:offset + size])

    return {
        "header": header,
        "reserved": [reserved_a, reserved_b],
        "entries": entries,
        "size": len(data),
    }

def build_dlc(entries, header=DEFAULT_HEADER, reserved=(0, 0)):
    if not 0 < len(entries) <= SLOT_COUNT:
        raise DlcError(f"{len(entries)} entries, a DLC holds {SLOT_COUNT}")

    table = bytearray(DATA_START)
    DLC_HEADER.pack_into(table, 0, header, len(entries), reserved[0], reserved[1])

    chunks = [table]
    position = DATA_START
    for index, entry in enumerate(entries):
        struct.pack_into("<I", table, OFFSET_TABLE + index * 4, position)
        struct.pack_into("<I", table, SIZE_TABLE + index * 4, len(entry))
        padded = align_up(len(entry))
        chunks.append(entry)
        chunks.append(bytes(padded - len(entry)))
        position += padded

    return b"".join(chunks)

def write_atomic(path, data):
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def format_size(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024

def unpack_folder(folder, log):
    folder = Path(folder)
    bins = sorted(
        (item for item in folder.iterdir() if item.is_file() and item.suffix.lower() == ".bin"),
        key=lambda item: item.name.lower(),
    )
    if not bins:
        log(f"No .bin files in {folder}", "warn")
        return

    out_root = folder.parent / f"{folder.name}_Unpacked"
    out_root.mkdir(exist_ok=True)
    log(f"Unpacking {len(bins)} DLC files into {out_root}", "info")

    done = skipped = failed = 0
    for bin_path in bins:
        out_dir = out_root / bin_path.stem
        if out_dir.exists() and any(out_dir.iterdir()):
            log(f"{bin_path.name}: {out_dir.name} already exists, delete it to unpack again", "warn")
            skipped += 1
            continue

        try:
            dlc = read_dlc(bin_path)
        except (DlcError, OSError) as exc:
            log(f"{bin_path.name}: not a DLC container, {exc}", "error")
            failed += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        names = []
        subs = 0
        for index, entry in enumerate(dlc["entries"]):
            name, entry_subs = write_entry(out_dir, index, entry)
            names.append(name)
            subs += entry_subs

        manifest = {
            "source": bin_path.name,
            "source_size": dlc["size"],
            "header": dlc["header"],
            "reserved": dlc["reserved"],
            "files": names,
        }
        (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        sub_note = f", {subs} sub containers" if subs else ""
        log(f"{bin_path.name}: {len(names)} entries{sub_note}, {format_size(dlc['size'])}", "good")
        done += 1

    summary = f"Unpack finished: {done} unpacked, {skipped} skipped, {failed} failed"
    log(summary, "good" if not failed else "warn")
    log(f"Output: {out_root}", "muted")

def load_manifest(folder):
    path = folder / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DlcError(f"{MANIFEST_NAME} couldnt be read, {exc}")
    return data if isinstance(data, dict) else {}

def collect_entries(folder):
    found = {}
    for item in folder.iterdir():
        if item.is_dir():
            if not (item / SUB_MANIFEST).exists():
                continue
        elif item.name in (MANIFEST_NAME, SUB_MANIFEST, SUB_STRINGS) or item.name.endswith(".tmp"):
            continue
        match = INDEX_PATTERN.match(item.name)
        if not match:
            continue
        index = int(match.group(1))
        if index in found:
            raise DlcError(f"{found[index].name} and {item.name} both claim entry {index}")
        found[index] = item

    if not found:
        return []

    expected = list(range(len(found)))
    if sorted(found) != expected:
        missing = sorted(set(range(max(found) + 1)) - set(found))
        raise DlcError(f"entry numbering has gaps, missing {', '.join(map(str, missing))}")
    return [found[index] for index in expected]

def is_dlc_folder(folder):
    if (folder / MANIFEST_NAME).exists():
        return True
    return any(
        item.is_file() and INDEX_PATTERN.match(item.name)
        for item in folder.iterdir()
    )

def rebuilt_root(base):
    name = base.name
    if name.lower().endswith("_unpacked"):
        name = name[:-len("_unpacked")]
    return base.parent / f"{name}_Rebuilt"

def rebuild_folder(folder, log):
    folder = Path(folder)
    if is_dlc_folder(folder):
        base, targets = folder.parent, [folder]
    else:
        base = folder
        targets = sorted(
            (item for item in folder.iterdir() if item.is_dir() and is_dlc_folder(item)),
            key=lambda item: item.name.lower(),
        )

    if not targets:
        log(f"No unpacked DLC folders found in {folder}", "warn")
        return

    out_root = rebuilt_root(base)
    out_root.mkdir(exist_ok=True)
    log(f"Rebuilding {len(targets)} DLC folders into {out_root}", "info")

    done = failed = 0
    for target in targets:
        try:
            manifest = load_manifest(target)
            files = collect_entries(target)
            if not files:
                raise DlcError("no numbered entry files found")

            expected_count = len(manifest.get("files", [])) or len(files)
            if len(files) != expected_count:
                log(
                    f"{target.name}: {len(files)} entries but the original had {expected_count}, "
                    "the game may not expect that",
                    "warn",
                )

            entries = [read_entry(path, log, target.name) for path in files]
            header = int(manifest.get("header", DEFAULT_HEADER))
            reserved = manifest.get("reserved", [0, 0])
            data = build_dlc(entries, header, reserved)

            check = read_dlc_bytes(data)
            if check != entries:
                raise DlcError("verification failed, rebuilt entries dont match the inputs")

            out_name = manifest.get("source") or f"{target.name}.bin"
            out_path = out_root / out_name
            write_atomic(out_path, data)

            original = manifest.get("source_size")
            if isinstance(original, int) and original != len(data):
                delta = len(data) - original
                sign = "+" if delta > 0 else "-"
                size_note = f"{format_size(len(data))} ({sign}{format_size(abs(delta))} vs original)"
            else:
                size_note = format_size(len(data))
            log(f"{out_name}: {len(entries)} entries, {size_note}", "good")
            done += 1
        except (DlcError, OSError, ValueError, TypeError) as exc:
            log(f"{target.name}: {exc}", "error")
            failed += 1

    summary = f"Rebuild finished: {done} rebuilt, {failed} failed"
    log(summary, "good" if not failed else "warn")
    log(f"Output: {out_root}", "muted")

def read_dlc_bytes(data):
    header, count, reserved_a, reserved_b = DLC_HEADER.unpack_from(data, 0)
    offsets = struct.unpack_from(f"<{count}I", data, OFFSET_TABLE)
    sizes = struct.unpack_from(f"<{count}I", data, SIZE_TABLE)
    return [data[offset:offset + size] for offset, size in zip(offsets, sizes)]

def build_offset_table(entries, align):
    table = bytearray(align_up(4 + 4 * len(entries), align))
    struct.pack_into("<I", table, 0, len(entries))
    chunks = [table]
    position = len(table)
    for index, entry in enumerate(entries):
        start = align_up(position, align)
        chunks.append(bytes(start - position))
        struct.pack_into("<I", table, 4 + index * 4, start)
        chunks.append(entry)
        position = start + len(entry)
    return b"".join(chunks)

def build_pair_table(entries, align):
    table = bytearray(4 + 8 * len(entries))
    struct.pack_into("<I", table, 0, len(entries))
    chunks = [table]
    position = len(table)
    for index, entry in enumerate(entries):
        start = align_up(position, align)
        chunks.append(bytes(start - position))
        struct.pack_into("<II", table, 4 + index * 8, start, len(entry))
        chunks.append(entry)
        position = start + len(entry)
    return b"".join(chunks)

def parse_offset_table(data):
    if len(data) < 12:
        return None
    count = struct.unpack_from("<I", data, 0)[0]
    head = 4 + 4 * count
    if not 2 <= count <= SUB_MAX_COUNT or head > len(data):
        return None
    offsets = list(struct.unpack_from(f"<{count}I", data, 4))
    if any(later <= earlier for earlier, later in zip(offsets, offsets[1:])) or offsets[-1] >= len(data):
        return None
    for align in SUB_ALIGNS:
        if offsets[0] != align_up(head, align) or any(offset % align for offset in offsets):
            continue
        bounds = offsets + [len(data)]
        entries = [data[bounds[index]:bounds[index + 1]] for index in range(count)]
        if build_offset_table(entries, align) == data:
            return align, entries
    return None

def parse_pair_table(data):
    if len(data) < 12:
        return None
    count = struct.unpack_from("<I", data, 0)[0]
    head = 4 + 8 * count
    if not 1 <= count <= SUB_MAX_COUNT or head > len(data):
        return None
    pairs = [struct.unpack_from("<II", data, 4 + index * 8) for index in range(count)]
    if pairs[-1][0] + pairs[-1][1] != len(data):
        return None
    for align in SUB_ALIGNS + (1,):
        position = head
        for offset, size in pairs:
            if offset != align_up(position, align):
                break
            position = offset + size
        else:
            entries = [data[offset:offset + size] for offset, size in pairs]
            if build_pair_table(entries, align) == data:
                return align, entries
    return None

def split_container(data):
    parsed = parse_offset_table(data)
    if parsed:
        return "offsets", parsed[0], parsed[1]
    parsed = parse_pair_table(data)
    if parsed:
        return "pairs", parsed[0], parsed[1]
    return None

def build_container(kind, align, entries):
    if kind == "offsets":
        return build_offset_table(entries, align)
    if kind == "pairs":
        return build_pair_table(entries, align)
    raise DlcError(f"unknown sub container kind {kind!r}")

def is_string_table(entries):
    for entry in entries:
        if not entry.endswith(b"\0") or b"\0" in entry[:-1]:
            return False
        try:
            entry[:-1].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True

def escape_string(text):
    return "".join(STRING_ESCAPES.get(char, char) for char in text)

def unescape_string(line):
    chars = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line) and line[index + 1] in STRING_UNESCAPES:
            chars.append(STRING_UNESCAPES[line[index + 1]])
            index += 2
            continue
        chars.append(char)
        index += 1
    return "".join(chars)

def strings_to_text(entries):
    return "\n".join(escape_string(entry[:-1].decode("utf-8")) for entry in entries) + "\n"

def text_to_strings(text):
    if text.endswith("\n"):
        text = text[:-1]
    return [unescape_string(line.rstrip("\r")).encode("utf-8") + b"\0" for line in text.split("\n")]

def write_entry(folder, index, data, depth=0):
    base = f"{index:03d}"
    split = split_container(data) if depth < SUB_MAX_DEPTH else None
    if not split:
        name = base + detect_ext(data)
        (folder / name).write_bytes(data)
        return name, 0

    kind, align, entries = split
    sub = folder / base
    sub.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": kind, "align": align, "count": len(entries)}
    subs = 1

    if is_string_table(entries) and text_to_strings(strings_to_text(entries)) == entries:
        (sub / SUB_STRINGS).write_text(strings_to_text(entries), encoding="utf-8", newline="\n")
        manifest["strings"] = SUB_STRINGS
    else:
        names = []
        for child_index, child in enumerate(entries):
            name, child_subs = write_entry(sub, child_index, child, depth + 1)
            names.append(name)
            subs += child_subs
        manifest["files"] = names

    (sub / SUB_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return base, subs

def load_sub_manifest(folder):
    try:
        data = json.loads((folder / SUB_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DlcError(f"{folder.name}/{SUB_MANIFEST} couldnt be read, {exc}")
    if not isinstance(data, dict) or "kind" not in data:
        raise DlcError(f"{folder.name}/{SUB_MANIFEST} is missing its kind")
    return data

def read_entry(path, log, label=""):
    label = f"{label}/{path.name}" if label else path.name
    if path.is_file():
        return path.read_bytes()

    manifest = load_sub_manifest(path)
    if manifest.get("strings"):
        text = (path / manifest["strings"]).read_text(encoding="utf-8-sig")
        entries = text_to_strings(text)
    else:
        entries = [read_entry(child, log, label) for child in collect_entries(path)]
        if not entries:
            raise DlcError(f"{label} holds no numbered entries")

    expected = manifest.get("count")
    if isinstance(expected, int) and expected != len(entries):
        log(f"{label}: {len(entries)} entries but the original had {expected}, the game may not expect that", "warn")
    return build_container(manifest["kind"], int(manifest.get("align", 4)), entries)

class VirtualLog(tk.Frame):
    def __init__(self, master, max_lines=200000, **kwargs):
        super().__init__(master, bg=BORDER, padx=1, pady=1, **kwargs)
        self.font = tkfont.Font(family="Consolas", size=9)
        self.line_height = self.font.metrics("linespace") + 3
        self.max_lines = max_lines
        self.lines = []
        self.first = 0
        self.follow = True
        self.render_pending = False
        self.items = []

        inner = tk.Frame(self, bg=BG_ALT)
        inner.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(inner, bg=BG_ALT, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(inner, orient="vertical", command=self.on_scrollbar)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Configure>", lambda event: self.clamp_and_render())
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-3>", self.copy_all)

    def visible_rows(self):
        return max(1, (self.canvas.winfo_height() - 8) // self.line_height)

    def max_first(self):
        return max(0, len(self.lines) - self.visible_rows())

    def append(self, entries):
        self.lines.extend(entries)
        overflow = len(self.lines) - self.max_lines
        if overflow > 0:
            del self.lines[:overflow]
            self.first = max(0, self.first - overflow)
        if self.follow:
            self.first = self.max_first()
        self.schedule_render()

    def scroll_to(self, index):
        self.first = min(max(0, index), self.max_first())
        self.follow = self.first >= self.max_first()
        self.schedule_render()

    def on_wheel(self, event):
        self.scroll_to(self.first - int(event.delta / 120) * 3)

    def on_scrollbar(self, action, amount, units=None):
        if action == "moveto":
            self.scroll_to(int(float(amount) * len(self.lines)))
        elif action == "scroll":
            step = self.visible_rows() if units == "pages" else 1
            self.scroll_to(self.first + int(amount) * step)

    def clamp_and_render(self):
        if self.follow:
            self.first = self.max_first()
        else:
            self.first = min(self.first, self.max_first())
        self.schedule_render()

    def schedule_render(self):
        if not self.render_pending:
            self.render_pending = True
            self.after_idle(self.render)

    def render(self):
        self.render_pending = False
        rows = self.visible_rows() + 1

        while len(self.items) < rows:
            row = len(self.items)
            self.items.append(self.canvas.create_text(
                8, 4 + row * self.line_height,
                anchor="nw", font=self.font, fill=TEXT, text="",
            ))
        for item in self.items[rows:]:
            self.canvas.itemconfigure(item, text="")

        for row in range(rows):
            index = self.first + row
            if index < len(self.lines):
                text, color = self.lines[index]
                self.canvas.itemconfigure(self.items[row], text=text, fill=color)
            else:
                self.canvas.itemconfigure(self.items[row], text="")

        total = len(self.lines)
        if total:
            self.scrollbar.set(self.first / total, min(1.0, (self.first + rows - 1) / total))
        else:
            self.scrollbar.set(0.0, 1.0)

    def copy_all(self, event=None):
        self.clipboard_clear()
        self.clipboard_append("\n".join(text for text, color in self.lines))
        self.append([("Log copied to clipboard", TEXT_MUTED)])

class HoverCard(tk.Canvas):
    def __init__(self, master, title, subtitle, command, color=ACCENT, width=320, height=108, **kwargs):
        super().__init__(master, width=width, height=height, bg=BG, highlightthickness=0, bd=0, **kwargs)
        self.command = command
        self.default_bg = PANEL_ALT
        self.hover_bg = "#3A2A21"
        self.disabled_bg = PANEL
        self.enabled = True

        self.rect = self.create_rectangle(6, 6, width - 6, height - 6, fill=self.default_bg, outline=color, width=2)
        self.accent = self.create_rectangle(10, 10, 20, height - 10, fill=color, outline=color)
        self.spark = self.create_line(
            width - 48, 12, width - 14, height - 12,
            fill=ACCENT_DEEP if color == ACCENT else color, width=3,
        )
        self.title_text = self.create_text(28, 34, text=title, font=("Segoe UI", 14, "bold"), anchor="w", fill=TEXT)
        self.sub_text = self.create_text(
            28, 63, text=subtitle, font=("Segoe UI", 9), anchor="w", fill=TEXT_MUTED, width=width - 66,
        )

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.itemconfig(self.rect, fill=self.default_bg if enabled else self.disabled_bg)
        self.itemconfig(self.title_text, fill=TEXT if enabled else TEXT_MUTED)
        self.config(cursor="")

    def on_enter(self, event=None):
        if self.enabled:
            self.itemconfig(self.rect, fill=self.hover_bg)
            self.config(cursor="hand2")

    def on_leave(self, event=None):
        if self.enabled:
            self.itemconfig(self.rect, fill=self.default_bg)
        self.config(cursor="")

    def on_click(self, event=None):
        if self.enabled:
            self.command()
        return "break"

def setup_styles(root):
    style = ttk.Style(master=root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "Vertical.TScrollbar",
        background=PANEL_SOFT,
        troughcolor=BG_ALT,
        bordercolor=PANEL,
        arrowcolor=ACCENT_BRIGHT,
        relief="flat",
    )
    style.map("Vertical.TScrollbar", background=[("active", ACCENT)])

class KatsukiTool:
    def __init__(self, root):
        self.root = root
        self.root.title("OP3 DLC Modding Software")
        self.root.geometry("800x780")
        self.root.minsize(720, 560)
        self.root.configure(bg=BG)
        setup_styles(self.root)

        self.messages = queue.Queue()
        self.busy = False
        self.last_folder = str(SCRIPT_DIR)
        self.root.report_callback_exception = self.report_callback_exception

        header = tk.Frame(root, bg=BG)
        header.pack(pady=(36, 24), fill="x")
        tk.Label(header, text="Katsuki's Software", font=("Impact", 36), bg=BG, fg=TEXT).pack()

        cards = tk.Frame(root, bg=BG)
        cards.pack()
        self.unpack_card = HoverCard(
            cards, title="Unpack DLCs", subtitle="Pick a folder of DLC .bin files to extract",
            command=self.start_unpack, color=METAL,
        )
        self.rebuild_card = HoverCard(
            cards, title="Rebuild DLCs", subtitle="Pick unpacked DLC folders to pack back into .bin",
            command=self.start_rebuild, color=ACCENT_DEEP,
        )
        self.cards = (self.unpack_card, self.rebuild_card)
        for index, card in enumerate(self.cards):
            card.grid(row=0, column=index, padx=8, pady=10)

        self.status = tk.Label(root, text="System fuckin ready", font=("Segoe UI", 9, "bold"), bg=BG, fg=TEXT)
        self.status.pack(pady=(6, 10))

        self.log_view = VirtualLog(root)
        self.log_view.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        self.log("Katsuki's Software is ready. Right click the log to copy it.", "muted")
        self.root.after(50, self.drain_messages)

    def report_callback_exception(self, exc_type, exc_value, exc_traceback):
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        log_path = write_crash_log(detail)
        self.log(f"Unexpected error: {exc_value}", "error")
        for line in detail.rstrip().splitlines():
            self.log(line, "error")
        if log_path:
            self.log(f"Saved to {log_path}", "muted")
        self.set_status("Something broke, see the log", ERROR_TEXT)

    def log(self, text, level="info"):
        stamp = time.strftime("%H:%M:%S")
        self.messages.put(("log", f"[{stamp}] {text}", level))

    def drain_messages(self):
        batch = []
        try:
            for count in range(5000):
                kind, payload, extra = self.messages.get_nowait()
                if kind == "log":
                    batch.append((payload, LEVEL_COLORS.get(extra, TEXT)))
                elif kind == "done":
                    self.finish_task(payload, extra)
        except queue.Empty:
            pass
        if batch:
            self.log_view.append(batch)
        self.root.after(50, self.drain_messages)

    def set_status(self, text, color=TEXT):
        self.status.config(text=text, fg=color)

    def set_busy(self, busy):
        self.busy = busy
        for card in self.cards:
            card.set_enabled(not busy)

    def pick_folder(self, title):
        folder = filedialog.askdirectory(title=title, initialdir=self.last_folder, mustexist=True)
        if folder:
            self.last_folder = folder
        return folder

    def start_unpack(self):
        folder = self.pick_folder("Select a folder of DLC .bin files")
        if folder:
            self.run_task("Unpack", unpack_folder, folder)

    def start_rebuild(self):
        folder = self.pick_folder("Select unpacked DLC folders to rebuild")
        if folder:
            self.run_task("Rebuild", rebuild_folder, folder)

    def run_task(self, name, task, folder):
        if self.busy:
            return
        self.set_busy(True)
        self.set_status(f"{name} runnin.", ACCENT_BRIGHT)
        self.log(f"{name}: {folder}", "info")

        def worker():
            ok = True
            try:
                task(folder, self.log)
            except Exception as exc:
                ok = False
                self.log(f"{name} crashed: {exc}", "error")
                for line in traceback.format_exc().rstrip().splitlines():
                    self.log(line, "error")
            self.messages.put(("done", name, ok))

        threading.Thread(target=worker, name=f"katsuki-{name.lower()}", daemon=True).start()

    def finish_task(self, name, ok):
        self.set_busy(False)
        if ok:
            self.set_status(f"{name} complete", GREEN_BRIGHT)
        else:
            self.set_status(f"{name} failed, see the log", ERROR_TEXT)

def thread_excepthook(args):
    write_crash_log("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

def main():
    threading.excepthook = thread_excepthook
    try:
        root = tk.Tk()
    except tk.TclError:
        fatal_exit(
            f"{APP_NAME} couldnt start Tk. The Python install's Tcl/Tk files look damaged or missing.\n\n"
            "Rerun the python.org installer and choose Repair.",
            traceback.format_exc(),
        )
    try:
        KatsukiTool(root)
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass
        fatal_exit(f"{APP_NAME} hit an error while building its window.", traceback.format_exc())
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        fatal_exit(f"{APP_NAME} closed because of an unexpected error.", traceback.format_exc())
