from __future__ import annotations

import hashlib, io, json, math, os, sys, traceback
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional, Sequence, Tuple


SCRIPT_PATH = os.path.abspath(sys.argv[0] or __file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
os.chdir(SCRIPT_DIR)

APP_TITLE = "WO3 Bubble Unit Editor"
BIN_FILENAME = "LINKFILE_000.bin"
BIN_FILENAME_UPPER = "LINKFILE_000.BIN"
UNIT_DATA_OFFSET = 0x44CDA
UNIT_SLOT_COUNT = 2206
UNIT_SLOT_SIZE = 40
UNIT_BLOCK_SIZE = UNIT_SLOT_COUNT * UNIT_SLOT_SIZE
MIXED_VALUE_TEXT = "Mixed Value"
FIELD_NAME_LIMIT = 40

APP_DATA_DIR = os.path.join(SCRIPT_DIR, "wo3_unit_editor")
MOD_DIR = os.path.join(APP_DATA_DIR, "mods")
BACKUP_DIR = os.path.join(APP_DATA_DIR, "backups")
FIELD_NAMES_JSON = os.path.join(APP_DATA_DIR, "field_names.json")
CRASH_LOG_PATH = os.path.join(APP_DATA_DIR, "wo3_unit_editor_crash.log")

BG = "#081124"
BG_ALT = "#111d3d"
PANEL = "#132347"
PANEL_ALT = "#1B2F5A"
PANEL_EDGE = "#3B63AB"
TEXT = "#F6FAFF"
SUBTEXT = "#B8C8EC"
MUTED = "#7E93C4"
STATUS_GOOD = "#8DF1BE"
STATUS_WARN = "#FFD87A"
STATUS_BAD = "#FF98A8"
FIELD_DECK = "#DDE7FF"
FIELD_CARD = "#F4F8FF"
FIELD_CARD_ALT = "#E9F0FF"
FIELD_TEXT = "#11203E"
FIELD_HINT = "#556A9E"
FIELD_BORDER = "#9BB5EA"
FIELD_INVALID = "#FFD4DE"
BUBBLE_GOLD = "#FFD864"
BUBBLE_GREEN = "#41E39B"
BUBBLE_BLUE = "#65AFFF"
BUBBLE_PINK = "#FF6DB5"
BUBBLE_PURPLE = "#B072FF"
HERO_LINE = "#4B71C4"
SHADOW = "#040916"

CTRL_MASK = 0x0004
SHIFT_MASK = 0x0001

BUBBLE_PALETTE = (
    {"fill": "#C252FF", "outline": "#7A2EB7", "shine": "#F3C4FF"},
    {"fill": "#396CFF", "outline": "#183C9A", "shine": "#B9CBFF"},
    {"fill": "#FFE23C", "outline": "#C59B00", "shine": "#FFF5A5"},
    {"fill": "#FF4B3A", "outline": "#A51E16", "shine": "#FFC2BD"},
    {"fill": "#38DD57", "outline": "#178733", "shine": "#BFF7CA"},
    {"fill": "#FF59C8", "outline": "#A91D7D", "shine": "#FFD0EF"},
)

TARGET_VISIBLE_BUBBLES = 108


@dataclass(frozen=True)
class FieldSpec:
    name: str
    size: int
    offset: int


def build_field_specs() -> List[FieldSpec]:
    layout = [
        ("Name", 2),
        ("Unknown 1", 1),
        ("Unknown 2", 1),

        ("Voice ID", 2),
        ("Model ID", 2),
        ("Moveset", 2),

        ("Unknown 6", 1),
        ("Unknown 7", 1),
        ("Unknown 8", 1),
        ("Unknown 9", 1),

        ("Life", 2),

        ("Unknown 10", 1),
        ("Unknown 11", 1),
        ("Unknown 12", 1),
        ("Unknown 13", 1),
        ("Unknown 14", 1),
        ("Unknown 15", 1),

        ("Param 1", 2),
        ("Param 2", 2),
        ("Jump", 2),
        ("Speed", 2),

        ("Unknown 16", 1),
        ("Unknown 17", 1),
        ("AI level?", 2),
        ("AI related?", 1),
        ("AI Type 2?", 1),
        ("Unknown 18", 1),
        ("Weapon ID", 2),
        ("Costume ID", 1),
    ]

    specs: List[FieldSpec] = []
    offset = 0
    for name, size in layout:
        specs.append(FieldSpec(name=name, size=size, offset=offset))
        offset += size

    if offset != UNIT_SLOT_SIZE:
        raise ValueError(f"Field layout totals {offset} bytes instead of {UNIT_SLOT_SIZE}.")

    return specs


FIELD_SPECS = build_field_specs()
DEFAULT_FIELD_NAMES = [spec.name for spec in FIELD_SPECS]


def ensure_app_dirs() -> None:
    os.makedirs(MOD_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def safe_relpath(path: Optional[str]) -> str:
    if not path:
        return "Not set"
    try:
        rel = os.path.relpath(path, SCRIPT_DIR)
        return rel if not rel.startswith("..") else path
    except ValueError:
        return path


def normalize_field_names(names: Sequence[str]) -> List[str]:
    cleaned = [str(name).strip()[:FIELD_NAME_LIMIT] for name in names]
    if len(cleaned) != len(DEFAULT_FIELD_NAMES):
        cleaned = []
    result = []
    for index, default in enumerate(DEFAULT_FIELD_NAMES):
        result.append(cleaned[index] if index < len(cleaned) and cleaned[index] else default)
    return result


def save_field_names_json(path: str, names: Sequence[str]) -> None:
    payload = {
        "field_names": normalize_field_names(names),
        "slot_size": UNIT_SLOT_SIZE,
        "slot_count": UNIT_SLOT_COUNT,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_field_names_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        source = payload.get("field_names", [])
    elif isinstance(payload, list):
        source = payload
    else:
        raise ValueError("Field name JSON must be an object with 'field_names' or a simple list.")
    if not isinstance(source, list):
        raise ValueError("Field name JSON must contain a list of names.")
    return normalize_field_names(source)


def rounded_rect_points(x1: float, y1: float, x2: float, y2: float, radius: float) -> List[float]:
    radius = max(1.0, min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    return [
        x1 + radius,
        y1,
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1 + radius,
        x1,
        y1,
        x1 + radius,
        y1,
    ]


def draw_round_rect(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs):
    return canvas.create_polygon(rounded_rect_points(x1, y1, x2, y2, radius), smooth=True, splinesteps=36, **kwargs)


def int_bits_for_size(size: int) -> int:
    return size * 8


def mask_for_size(size: int) -> int:
    return (1 << int_bits_for_size(size)) - 1


def unsigned_to_signed(value: int, size: int) -> int:
    bits = int_bits_for_size(size)
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def parse_field_text(text: str, size: int) -> int:
    raw = (text or "").strip().replace("_", "")
    if not raw:
        raise ValueError("Value cannot be empty.")
    if raw.lower().startswith(("0x", "-0x", "+0x")):
        raise ValueError("Use an unsigned decimal integer.")
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError("Use an unsigned decimal integer.") from exc
    bits = int_bits_for_size(size)
    unsigned_max = (1 << bits) - 1
    if value < 0 or value > unsigned_max:
        raise ValueError(f"Value must stay within unsigned {bits}-bit range.")
    return value & mask_for_size(size)


def format_field_value(field: FieldSpec, value: int) -> str:
    return str(value)


def helper_text_for_value(field: FieldSpec, value: int) -> str:
    bits = int_bits_for_size(field.size)
    return f"u{bits} {value}"


def slot_offset_in_block(slot_index: int) -> int:
    return slot_index * UNIT_SLOT_SIZE


def slot_offset_in_bin(slot_index: int) -> int:
    return UNIT_DATA_OFFSET + slot_offset_in_block(slot_index)


def read_unit_block_from_bin(path: str) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        end = UNIT_DATA_OFFSET + UNIT_BLOCK_SIZE
        if size < end:
            raise ValueError(
                f"{os.path.basename(path)} is only {size} bytes, but the unit block ends at 0x{end:X}."
            )
        handle.seek(UNIT_DATA_OFFSET)
        blob = handle.read(UNIT_BLOCK_SIZE)
    if len(blob) != UNIT_BLOCK_SIZE:
        raise ValueError(f"Could not read the full {UNIT_BLOCK_SIZE} byte unit block from {os.path.basename(path)}.")
    return blob


def write_unit_block_to_bin(path: str, block: bytes) -> None:
    if len(block) != UNIT_BLOCK_SIZE:
        raise ValueError(f"Expected a {UNIT_BLOCK_SIZE} byte block, got {len(block)} bytes.")
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        end = UNIT_DATA_OFFSET + UNIT_BLOCK_SIZE
        if size < end:
            raise ValueError(
                f"{os.path.basename(path)} is only {size} bytes, but the unit block ends at 0x{end:X}."
            )
        handle.seek(UNIT_DATA_OFFSET)
        handle.write(block)


def read_slot_values(block: bytes, slot_index: int) -> List[int]:
    base = slot_offset_in_block(slot_index)
    values: List[int] = []
    for field in FIELD_SPECS:
        start = base + field.offset
        values.append(int.from_bytes(block[start : start + field.size], "little", signed=False))
    return values


def summarize_ranges(slots: Sequence[int], *, limit: int = 6) -> str:
    if not slots:
        return "None"
    sorted_slots = sorted(dict.fromkeys(slots))
    groups: List[str] = []
    start = prev = sorted_slots[0]
    for slot in sorted_slots[1:]:
        if slot == prev + 1:
            prev = slot
            continue
        groups.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = slot
    groups.append(f"{start}" if start == prev else f"{start}-{prev}")
    if len(groups) <= limit:
        return ", ".join(groups)
    return ", ".join(groups[:limit]) + f", +{len(groups) - limit} more"


def parse_slot_expression(text: str) -> List[int]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Enter a slot or range like 128 or 128-140.")
    slots = set()
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            try:
                start = int(left.strip(), 0)
                end = int(right.strip(), 0)
            except ValueError as exc:
                raise ValueError(f"Could not read range '{piece}'.") from exc
            if start <= end:
                span = range(start, end + 1)
            else:
                span = range(start, end - 1, -1)
            for slot in span:
                if slot < 0 or slot >= UNIT_SLOT_COUNT:
                    raise ValueError(f"Slot {slot} is outside 0-{UNIT_SLOT_COUNT - 1}.")
                slots.add(slot)
        else:
            try:
                slot = int(piece, 0)
            except ValueError as exc:
                raise ValueError(f"Could not read slot '{piece}'.") from exc
            if slot < 0 or slot >= UNIT_SLOT_COUNT:
                raise ValueError(f"Slot {slot} is outside 0-{UNIT_SLOT_COUNT - 1}.")
            slots.add(slot)
    if not slots:
        raise ValueError("No slots were found in that expression.")
    return sorted(slots)


def backup_path_for_bin(path: str) -> str:
    token = hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(BACKUP_DIR, f"{stem}_{token}_unit_backup.bin")


def find_candidate_bins() -> List[str]:
    candidates = []
    for filename in (BIN_FILENAME, BIN_FILENAME_UPPER):
        path = os.path.join(SCRIPT_DIR, filename)
        if os.path.isfile(path):
            candidates.append(path)
    return candidates


def pretty_file_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def read_mod_block(path: str) -> bytes:
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) != UNIT_BLOCK_SIZE:
        raise ValueError(
            f"{os.path.basename(path)} is {len(blob)} bytes, but a WO3 unit block mod must be {UNIT_BLOCK_SIZE} bytes."
        )
    return blob


def write_mod_block(path: str, block: bytes) -> None:
    if len(block) != UNIT_BLOCK_SIZE:
        raise ValueError(f"Expected a {UNIT_BLOCK_SIZE} byte block, got {len(block)} bytes.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(block)


class GlowButton(tk.Canvas):
    def __init__(
        self,
        parent: tk.Misc,
        text: str,
        command,
        *,
        fill: str,
        outline: str,
        glow: str,
        text_fill: str = TEXT,
        height: int = 50,
    ):
        super().__init__(parent, height=height, bg=parent.cget("bg"), highlightthickness=0, bd=0, relief="flat", cursor="hand2")
        self._text = text
        self._command = command
        self._fill = fill
        self._outline = outline
        self._glow = glow
        self._text_fill = text_fill
        self._enabled = True
        self._hover = False
        self._pressed = False
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.redraw()

    def set_text(self, text: str) -> None:
        self._text = text
        self.redraw()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self.redraw()

    def on_enter(self, _event) -> None:
        self._hover = True
        self.redraw()

    def on_leave(self, _event) -> None:
        self._hover = False
        self._pressed = False
        self.redraw()

    def on_press(self, _event) -> None:
        if self._enabled:
            self._pressed = True
            self.redraw()

    def on_release(self, event) -> None:
        if not self._enabled:
            return
        pressed = self._pressed
        self._pressed = False
        self.redraw()
        if pressed and self.find_overlapping(event.x, event.y, event.x, event.y) and callable(self._command):
            self._command()

    def redraw(self) -> None:
        self.delete("all")
        width = max(10, self.winfo_width())
        height = max(10, self.winfo_height())
        fill = self._fill
        outline = self._outline
        glow = self._glow
        text_fill = self._text_fill
        if not self._enabled:
            fill = "#394867"
            outline = "#4F5D7E"
            glow = "#394867"
            text_fill = "#A3B0CF"
        elif self._pressed:
            fill = outline
        elif self._hover:
            outline = glow
        draw_round_rect(self, 2, 3, width - 3, height - 3, 12, fill=fill, outline=outline, width=2 if self._enabled else 1)
        self.create_rectangle(10, 9, 14, height - 10, fill=outline, outline="")
        self.create_text(width / 2, height / 2, text=self._text, fill=text_fill, font=("Segoe UI", 10, "bold"))


class ToolbarCanvas(tk.Canvas):
    def __init__(self, parent: tk.Misc, controller: "WO3UnitEditor"):
        super().__init__(parent, height=126, bg=BG, highlightthickness=0, bd=0, relief="flat")
        self.controller = controller
        self.hover_key: Optional[str] = None
        self.button_boxes: List[Dict[str, object]] = []
        self.range_entry = tk.Entry(
            self,
            textvariable=controller.selection_entry_var,
            relief="flat",
            bd=0,
            bg=FIELD_CARD,
            fg=FIELD_TEXT,
            insertbackground=FIELD_TEXT,
            font=("Consolas", 10),
        )
        self.range_entry.bind("<Return>", lambda _e: controller.select_range_from_entry())
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Motion>", self.on_motion)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def button_specs(self) -> Tuple[List[Tuple[str, str, object, bool, str]], List[Tuple[str, str, object, bool, str]]]:
        ctrl = self.controller
        top = [
            ("open_bin", "Open BIN", ctrl.open_bin_file, True, BUBBLE_BLUE),
            ("load_mod", "Load Mod", ctrl.open_mod_file, True, BUBBLE_PURPLE),
            ("save_mod", "Save Mod As", ctrl.save_current_mod_as, ctrl.working_block_loaded(), BUBBLE_GREEN),
            ("load_saved", "Load Saved", ctrl.load_selected_library_mod, bool(ctrl.selected_library_path), BUBBLE_GOLD),
            ("apply_current", "Apply Current", ctrl.apply_current_to_bin, bool(ctrl.current_bin_path and ctrl.working_block_loaded()), BUBBLE_GOLD),
            ("apply_saved", "Apply Saved", ctrl.apply_selected_library_mod_to_bin, bool(ctrl.current_bin_path and ctrl.selected_library_path), BUBBLE_PINK),
            ("restore_bin", "Restore Backup", ctrl.restore_bin_from_backup, bool(ctrl.current_bin_path and ctrl.backup_block), BUBBLE_GREEN),
        ]
        bottom = [
            ("select_range", "Select Range", ctrl.select_range_from_entry, True, BUBBLE_BLUE),
            ("clear", "Clear", ctrl.clear_selection, bool(ctrl.selected_slots), BUBBLE_PURPLE),
            ("invert", "Invert", ctrl.invert_selection, ctrl.working_block_loaded(), BUBBLE_PINK),
            ("select_changed", "Changed", ctrl.select_changed_slots, bool(ctrl.changed_slots_vs_compare), BUBBLE_GREEN),
            ("prev_mod", "Prev Mod", lambda: ctrl.cycle_saved_mod(-1), bool(ctrl.mod_library_paths), BUBBLE_GOLD),
            ("next_mod", "Next Mod", lambda: ctrl.cycle_saved_mod(1), bool(ctrl.mod_library_paths), BUBBLE_GOLD),
            ("refresh_source", "Refresh", ctrl.refresh_memory_from_source, bool(ctrl.current_bin_path or ctrl.current_mod_path), BUBBLE_BLUE),
        ]
        return top, bottom

    def draw_button(self, x1: float, y1: float, x2: float, y2: float, text: str, accent: str, *, enabled: bool, hovered: bool) -> None:
        fill = PANEL_ALT if enabled else "#22324E"
        outline = accent if enabled else "#485977"
        text_fill = TEXT if enabled else "#8DA1C9"
        if hovered and enabled:
            fill = "#243B71"
        draw_round_rect(self, x1, y1, x2, y2, 12, fill=fill, outline=outline, width=2 if enabled else 1)
        self.create_rectangle(x1 + 8, y1 + 8, x1 + 12, y2 - 8, fill=outline, outline="")
        self.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill=text_fill, font=("Segoe UI", 9, "bold"))

    def redraw(self) -> None:
        self.delete("all")
        self.button_boxes.clear()
        width = max(320, self.winfo_width())
        height = max(100, self.winfo_height())
        self.create_rectangle(0, 0, width, height, fill=BG, outline="")
        draw_round_rect(self, 0, 0, width - 2, height - 2, 16, fill=BG_ALT, outline=PANEL_EDGE, width=1)
        for idx in range(4):
            y = 16 + (idx * 22)
            sway = 8 if idx % 2 else -8
            self.create_line(14, y, width - 14, y + sway, fill=HERO_LINE, width=1)

        top, bottom = self.button_specs()
        margin = 12
        gap = 8
        row_h = 30
        top_y = 14
        col_w = (width - (margin * 2) - (gap * (len(top) - 1))) / max(1, len(top))
        for idx, (key, text, command, enabled, accent) in enumerate(top):
            x1 = margin + idx * (col_w + gap)
            x2 = x1 + col_w
            hovered = self.hover_key == key
            self.draw_button(x1, top_y, x2, top_y + row_h, text, accent, enabled=bool(enabled), hovered=hovered)
            self.button_boxes.append({"key": key, "bbox": (x1, top_y, x2, top_y + row_h), "command": command, "enabled": bool(enabled)})

        bottom_y = top_y + row_h + 10
        entry_w = max(170.0, min(230.0, width * 0.17))
        draw_round_rect(self, margin, bottom_y, margin + entry_w, bottom_y + row_h, 12, fill=FIELD_CARD, outline=FIELD_BORDER, width=2)
        self.create_text(margin + 12, bottom_y + 7, anchor="nw", text="Slot / Range", fill=FIELD_HINT, font=("Segoe UI", 7, "bold"))
        self.create_window(margin + 86, bottom_y + (row_h / 2), anchor="w", width=max(70, entry_w - 98), height=20, window=self.range_entry)

        remain_x = margin + entry_w + gap
        col_w = (width - remain_x - margin - (gap * (len(bottom) - 1))) / max(1, len(bottom))
        for idx, (key, text, command, enabled, accent) in enumerate(bottom):
            x1 = remain_x + idx * (col_w + gap)
            x2 = x1 + col_w
            hovered = self.hover_key == key
            self.draw_button(x1, bottom_y, x2, bottom_y + row_h, text, accent, enabled=bool(enabled), hovered=hovered)
            self.button_boxes.append({"key": key, "bbox": (x1, bottom_y, x2, bottom_y + row_h), "command": command, "enabled": bool(enabled)})

        saved_mod = safe_relpath(self.controller.selected_library_path) if self.controller.selected_library_path else "None"
        canvas_text = [
            f"BIN: {safe_relpath(self.controller.current_bin_path)}",
            f"Backup: {safe_relpath(self.controller.backup_path)}",
            f"Current: {safe_relpath(self.controller.current_mod_path)}",
            f"Saved mod: {saved_mod}",
        ]
        self.create_text(14, height - 10, anchor="sw", text=" | ".join(canvas_text), fill=MUTED, font=("Consolas", 8))

    def find_button(self, x: float, y: float) -> Optional[Dict[str, object]]:
        for button in self.button_boxes:
            x1, y1, x2, y2 = button["bbox"]  # type: ignore[index]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return button
        return None

    def on_motion(self, event) -> None:
        button = self.find_button(event.x, event.y)
        hover_key = button["key"] if button else None  # type: ignore[index]
        if hover_key != self.hover_key:
            self.hover_key = hover_key
            self.configure(cursor="hand2" if button and button["enabled"] else "arrow")  # type: ignore[index]
            self.redraw()

    def on_leave(self, _event) -> None:
        self.hover_key = None
        self.configure(cursor="arrow")
        self.redraw()

    def on_click(self, event) -> None:
        button = self.find_button(event.x, event.y)
        if not button or not button["enabled"]:  # type: ignore[index]
            return
        command = button["command"]  # type: ignore[index]
        if callable(command):
            command()


class BubbleScrollbar(tk.Canvas):
    def __init__(self, parent: tk.Misc, *, bubble_fill: str = "#BF98D9"):
        super().__init__(parent, width=40, bg=parent.cget("bg"), highlightthickness=0, bd=0, relief="flat", cursor="hand2")
        self.bubble_fill = bubble_fill
        self._command = None
        self._first = 0.0
        self._last = 1.0
        self._dragging = False
        self._drag_offset = 0.0
        self._bubble_bbox: Optional[Tuple[float, float, float, float]] = None
        self._bubble_center_y = 0.0
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Leave>", self.on_leave)

    def set_command(self, command) -> None:
        self._command = command

    def set(self, first, last) -> None:
        self._first = float(first)
        self._last = float(last)
        self.redraw()

    def metrics(self) -> Optional[Tuple[float, float, float, float]]:
        height = max(80, self.winfo_height())
        if self._last - self._first >= 0.999:
            self._bubble_bbox = None
            return None
        radius = 14.0
        x = max(radius + 4, self.winfo_width() / 2.0)
        top = 10.0
        bottom = max(top + 30.0, height - 10.0)
        travel = max(1.0, bottom - top - (radius * 2.0))
        center_y = top + radius + (travel * clamp(self._first, 0.0, 1.0))
        self._bubble_center_y = center_y
        self._bubble_bbox = (x - radius, center_y - radius, x + radius, center_y + radius)
        return x, center_y, radius, travel

    def move_to(self, center_y: float) -> None:
        if self._command is None:
            return
        metrics = self.metrics()
        if metrics is None:
            return
        _, _, radius, travel = metrics
        top_center = 10.0 + radius
        clamped_y = clamp(center_y, top_center, top_center + travel)
        fraction = (clamped_y - top_center) / max(1.0, travel)
        self._command(fraction)

    def on_press(self, event) -> None:
        metrics = self.metrics()
        if metrics is None or self._bubble_bbox is None:
            return
        x1, y1, x2, y2 = self._bubble_bbox
        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            self._dragging = True
            self._drag_offset = event.y - self._bubble_center_y

    def on_drag(self, event) -> None:
        if not self._dragging:
            return
        self.move_to(event.y - self._drag_offset)

    def on_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        offset = self._drag_offset
        self._drag_offset = 0.0
        self.move_to(event.y - offset)

    def on_leave(self, _event) -> None:
        if not self._dragging:
            self.configure(cursor="hand2")

    def redraw(self) -> None:
        self.delete("all")
        metrics = self.metrics()
        if metrics is None or self._bubble_bbox is None:
            return
        x1, y1, x2, y2 = self._bubble_bbox
        radius = (x2 - x1) / 2.0
        self.create_oval(x1 + 2, y1 + 4, x2 + 2, y2 + 4, fill=SHADOW, outline="")
        self.create_oval(x1, y1, x2, y2, fill=self.bubble_fill, outline="#7D56A2", width=2)
        self.create_oval(
            x1 + (radius * 0.18),
            y1 + (radius * 0.18),
            x1 + (radius * 1.00),
            y1 + (radius * 1.00),
            fill="#EAD8F6",
            outline="",
            stipple="gray50",
        )
        self.create_oval(
            x1 + (radius * 0.60),
            y1 + (radius * 0.28),
            x1 + (radius * 0.88),
            y1 + (radius * 0.56),
            fill="#FFFFFF",
            outline="",
        )


class FieldCard(tk.Canvas):
    def __init__(self, parent: tk.Misc, controller: "WO3UnitEditor", field_index: int, field: FieldSpec):
        super().__init__(parent, height=28, bg=parent.cget("bg"), highlightthickness=0, bd=0, relief="flat")
        self.controller = controller
        self.field_index = field_index
        self.field = field
        self.name_var = tk.StringVar(value=self.controller.field_display_name(field_index))
        self.var = tk.StringVar(value="")
        self.name_entry = tk.Entry(
            self,
            textvariable=self.name_var,
            relief="flat",
            bd=0,
            bg=parent.cget("bg"),
            fg=FIELD_TEXT,
            insertbackground=FIELD_TEXT,
            font=("Segoe UI", 8, "bold"),
        )
        self.entry = tk.Entry(
            self,
            textvariable=self.var,
            relief="flat",
            bd=0,
            bg="#F3F6FF",
            fg=FIELD_TEXT,
            insertbackground=FIELD_TEXT,
            font=("Consolas", 9),
            disabledbackground="#CFD9F6",
            disabledforeground="#6D7EA8",
            justify="center",
        )
        self.name_window: Optional[int] = None
        self.entry_window: Optional[int] = None
        self.loaded_mixed = False
        self.user_edited = False
        self.enabled = False
        self._internal_update = False
        self.var.trace_add("write", self.on_var_write)
        self.entry.bind("<FocusIn>", self.on_focus_in)
        self.name_entry.bind("<MouseWheel>", self.delegate_wheel)
        self.entry.bind("<MouseWheel>", self.delegate_wheel)
        self.bind("<MouseWheel>", self.delegate_wheel)
        self.bind("<Configure>", lambda _e: self.redraw())
        self.redraw()

    def delegate_wheel(self, event):
        return self.controller.on_fields_mousewheel(event)

    def on_focus_in(self, _event) -> None:
        if self.loaded_mixed and not self.user_edited and self.var.get() == MIXED_VALUE_TEXT:
            try:
                self.entry.selection_range(0, "end")
            except tk.TclError:
                pass

    def on_var_write(self, *_args) -> None:
        if self._internal_update:
            return
        self.user_edited = True
        self.redraw()
        self.controller.on_field_card_changed(self)

    def current_field_name(self) -> str:
        return self.name_var.get().strip()[:FIELD_NAME_LIMIT]

    def set_field_name(self, text: str) -> None:
        self.name_var.set(text)
        self.redraw()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.entry.configure(state="normal" if enabled else "disabled")
        self.redraw()

    def set_display(self, text: str, *, mixed: bool = False) -> None:
        self._internal_update = True
        try:
            self.var.set(text)
        finally:
            self._internal_update = False
        self.loaded_mixed = mixed
        self.user_edited = False
        self.set_enabled(self.controller.has_selection())
        self.redraw()

    def current_value(self) -> Optional[int]:
        text = self.var.get()
        if self.loaded_mixed and not self.user_edited and text == MIXED_VALUE_TEXT:
            return None
        return parse_field_text(text, self.field.size)

    def helper_text(self) -> Tuple[str, str]:
        if not self.enabled:
            return ("Select at least one slot to edit this field.", MUTED)
        raw = self.var.get().strip()
        if self.loaded_mixed and not self.user_edited and raw == MIXED_VALUE_TEXT:
            return ("Selected slots disagree here. Type a value to override all of them.", FIELD_HINT)
        try:
            value = parse_field_text(raw, self.field.size)
        except ValueError:
            return ("Invalid value. Use an unsigned decimal integer.", STATUS_BAD)
        return (helper_text_for_value(self.field, value), FIELD_HINT)

    def accent_color(self) -> str:
        if self.field.name == "Name":
            return BUBBLE_GOLD
        if self.field.name == "Voice ID":
            return BUBBLE_BLUE
        if self.field.name == "Model ID":
            return BUBBLE_GREEN
        if self.field.name == "Moveset":
            return BUBBLE_PINK
        return BUBBLE_PURPLE

    def redraw(self) -> None:
        self.delete("all")
        width = max(84, self.winfo_width())
        height = max(24, self.winfo_height())
        _, helper_color = self.helper_text()
        if not self.enabled:
            row_fill = "#CED8EE"
            entry_bg = "#D7E0F3"
            entry_fg = "#6D7EA8"
            row_outline = FIELD_BORDER
            entry_outline = FIELD_BORDER
        elif helper_color == STATUS_BAD:
            row_fill = "#E6B6BF"
            entry_bg = "#F1C8CF"
            entry_fg = FIELD_TEXT
            row_outline = STATUS_BAD
            entry_outline = STATUS_BAD
        else:
            row_fill = FIELD_CARD if (self.field.offset // 2) % 2 == 0 else FIELD_CARD_ALT
            entry_bg = "#F4F7FF"
            entry_fg = MUTED if (self.loaded_mixed and not self.user_edited and self.var.get() == MIXED_VALUE_TEXT) else FIELD_TEXT
            row_outline = FIELD_BORDER
            entry_outline = self.accent_color() if self.loaded_mixed and not self.user_edited else FIELD_BORDER
        name_bg = row_fill
        name_fg = FIELD_TEXT
        self.entry.configure(bg=entry_bg, fg=entry_fg, insertbackground=FIELD_TEXT)
        self.name_entry.configure(bg=name_bg, fg=name_fg, insertbackground=FIELD_TEXT)
        draw_round_rect(self, 1, 2, width - 2, height - 2, 7, fill=row_fill, outline=row_outline, width=1)
        entry_w = min(96, max(72, width * 0.44))
        name_w = max(58, width - entry_w - 19)
        self.name_window = self.create_window(8, height / 2, anchor="w", width=name_w, height=18, window=self.name_entry)
        entry_x = width - entry_w - 7
        draw_round_rect(self, entry_x - 3, 4, width - 6, height - 4, 6, fill=entry_bg, outline=entry_outline, width=1)
        self.entry_window = self.create_window(entry_x, height / 2, anchor="w", width=entry_w - 6, height=18, window=self.entry)


class BubbleGridCanvas(tk.Canvas):
    def __init__(self, parent: tk.Misc, controller: "WO3UnitEditor"):
        super().__init__(parent, bg=BG, highlightthickness=0, bd=0, relief="flat")
        self.controller = controller
        self.item_to_slot: Dict[int, int] = {}
        self.scroll_y = 0.0
        self.total_height = 0.0
        self.columns = 1
        self.radius = 15
        self.x_step = 27
        self.y_step = 24
        self.pad_x = 28
        self.pad_right = 86
        self.pad_y = 98
        self.hover_slot: Optional[int] = None
        self.drag_start: Optional[Tuple[float, float]] = None
        self.drag_current: Optional[Tuple[float, float]] = None
        self.dragging = False
        self.drag_modifiers = 0
        self.scroll_bubble_bbox: Optional[Tuple[float, float, float, float]] = None
        self.scroll_bubble_center_y = 0.0
        self.scroll_bubble_top = 0.0
        self.scroll_bubble_bottom = 0.0
        self.scroll_bubble_radius = 0.0
        self.scroll_dragging = False
        self.scroll_drag_offset = 0.0
        self.bind("<Configure>", lambda _e: self.render())
        self.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<Button-4>", lambda _e: self.scroll_by(-80))
        self.bind("<Button-5>", lambda _e: self.scroll_by(80))
        self.bind("<Motion>", self.on_motion)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)

    def contentmetrics(self) -> Tuple[int, float]:
        width = max(280, self.winfo_width())
        height = max(220, self.winfo_height())
        available_w = max(180.0, width - self.pad_x - self.pad_right)
        available_h = max(120.0, height - self.pad_y - 24)
        approx_radius = math.sqrt((available_w * available_h) / max(1.0, TARGET_VISIBLE_BUBBLES * 2.9))
        radius = int(clamp(approx_radius, 22.0, 58.0))
        x_step = radius * 1.82
        y_step = radius * 1.56
        columns = max(6, int(available_w / max(10.0, x_step)))
        rows = math.ceil(UNIT_SLOT_COUNT / columns)
        chosen = (radius, x_step, y_step, columns, rows)
        self.radius, self.x_step, self.y_step, self.columns, rows = chosen
        self.total_height = self.pad_y + (rows * self.y_step) + (self.radius * 3.0) + 20
        max_scroll = max(0.0, self.total_height - max(1, self.winfo_height()))
        self.scroll_y = clamp(self.scroll_y, 0.0, max_scroll)
        return rows, max_scroll

    def slot_position(self, slot_index: int) -> Tuple[float, float]:
        row = slot_index // self.columns
        local_col = slot_index % self.columns
        river_shift = math.sin(row * 0.55) * (self.radius * 0.82)
        x = self.pad_x + self.radius + (local_col * self.x_step) + river_shift + ((self.x_step / 2.0) if row % 2 else 0)
        y = self.pad_y + self.radius + (row * self.y_step) - self.scroll_y
        return x, y

    def bubble_palette_index(self, slot_index: int) -> int:
        row = slot_index // self.columns
        col = slot_index % self.columns
        center = (self.columns - 1) / 2.0
        band = int(abs(col - center) / max(1.0, self.columns / 9.0))
        pocket = 4 if ((row + 2) % 9 in (6, 7) and (col % 11) in (2, 3, 4)) else None
        if pocket is not None:
            return pocket
        return (row // 2 + band) % len(BUBBLE_PALETTE)

    def visible(self, y: float) -> bool:
        return -32 <= y <= self.winfo_height() + 32

    def scroll_by(self, amount: float) -> None:
        self.contentmetrics()
        max_scroll = max(0.0, self.total_height - max(1, self.winfo_height()))
        self.scroll_y = clamp(self.scroll_y + amount, 0.0, max_scroll)
        self.render()

    def on_mousewheel(self, event):
        delta = -1 if event.delta > 0 else 1
        self.scroll_by(delta * 70)
        return "break"

    def slot_at(self, x: float, y: float) -> Optional[int]:
        hits = self.find_overlapping(x - 4, y - 4, x + 4, y + 4)
        for item_id in reversed(hits):
            slot = self.item_to_slot.get(item_id)
            if slot is not None:
                return slot
        return None

    def point_in_scroll_bubble(self, x: float, y: float) -> bool:
        if self.scroll_bubble_bbox is None:
            return False
        x1, y1, x2, y2 = self.scroll_bubble_bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def bubble_scrollmetrics(self, width: float, height: float, max_scroll: float) -> bool:
        if max_scroll <= 0:
            self.scroll_bubble_bbox = None
            return False
        radius = clamp(self.radius * 0.48, 16.0, 24.0)
        center_x = width - radius - 16
        top = self.pad_y + 10
        bottom = max(top + 40, height - radius - 18)
        travel = max(1.0, bottom - top - (radius * 2))
        ratio = 0.0 if max_scroll <= 0 else (self.scroll_y / max_scroll)
        center_y = top + radius + (travel * ratio)
        self.scroll_bubble_radius = radius
        self.scroll_bubble_center_y = center_y
        self.scroll_bubble_top = top + radius
        self.scroll_bubble_bottom = top + radius + travel
        self.scroll_bubble_bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        return True

    def set_scroll_from_bubble_center(self, center_y: float) -> None:
        max_scroll = max(0.0, self.total_height - max(1, self.winfo_height()))
        if max_scroll <= 0:
            return
        travel = max(1.0, self.scroll_bubble_bottom - self.scroll_bubble_top)
        clamped = clamp(center_y, self.scroll_bubble_top, self.scroll_bubble_bottom)
        ratio = (clamped - self.scroll_bubble_top) / travel
        self.scroll_y = ratio * max_scroll
        self.render()

    def on_motion(self, event) -> None:
        if self.scroll_dragging:
            self.set_scroll_from_bubble_center(event.y - self.scroll_drag_offset)
            return
        if self.dragging:
            return
        if self.point_in_scroll_bubble(event.x, event.y):
            self.hover_slot = None
            self.configure(cursor="hand2")
            return
        slot = self.slot_at(event.x, event.y)
        if slot != self.hover_slot:
            self.hover_slot = slot
            self.configure(cursor="hand2" if slot is not None else "arrow")

    def on_leave(self, _event) -> None:
        if self.scroll_dragging:
            return
        self.hover_slot = None
        self.configure(cursor="arrow")

    def on_press(self, event) -> None:
        if self.point_in_scroll_bubble(event.x, event.y):
            self.scroll_dragging = True
            self.scroll_drag_offset = event.y - self.scroll_bubble_center_y
            self.drag_start = None
            self.drag_current = None
            self.dragging = False
            self.configure(cursor="fleur")
            return
        self.drag_start = (event.x, event.y)
        self.drag_current = (event.x, event.y)
        self.dragging = False
        self.drag_modifiers = event.state

    def on_drag(self, event) -> None:
        if self.scroll_dragging:
            self.set_scroll_from_bubble_center(event.y - self.scroll_drag_offset)
            return
        if not self.drag_start:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        if not self.dragging and ((dx * dx) + (dy * dy)) >= 36:
            self.dragging = True
        if self.dragging:
            self.drag_current = (event.x, event.y)
            self.render()

    def on_release(self, event) -> None:
        if self.scroll_dragging:
            self.set_scroll_from_bubble_center(event.y - self.scroll_drag_offset)
            self.scroll_dragging = False
            self.scroll_drag_offset = 0.0
            self.configure(cursor="hand2" if self.point_in_scroll_bubble(event.x, event.y) else "arrow")
            return
        start = self.drag_start
        if not start:
            return
        if self.dragging:
            end = (event.x, event.y)
            self.dragging = False
            self.drag_start = None
            self.drag_current = None
            changed = self.finish_marquee_selection(start, end, self.drag_modifiers)
            if not changed:
                self.render()
            return
        self.drag_start = None
        self.drag_current = None
        slot = self.slot_at(event.x, event.y)
        self.controller.handle_bubble_click(slot, modifiers=event.state)

    def finish_marquee_selection(self, start: Tuple[float, float], end: Tuple[float, float], modifiers: int) -> bool:
        x1, y1 = start
        x2, y2 = end
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        selected = []
        for slot_index in range(UNIT_SLOT_COUNT):
            px, py = self.slot_position(slot_index)
            if left <= px <= right and top <= py <= bottom:
                selected.append(slot_index)
        if not selected:
            return False
        additive = bool(modifiers & (CTRL_MASK | SHIFT_MASK))
        if additive:
            self.controller.merge_selected_slots(selected, primary_slot=selected[-1])
        else:
            self.controller.set_selected_slots(selected, primary_slot=selected[-1])
        return True

    def render(self) -> None:
        self.delete("all")
        self.item_to_slot.clear()
        self.scroll_bubble_bbox = None
        rows, max_scroll = self.contentmetrics()
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self.create_rectangle(0, 0, width, height, fill=BG, outline="")
        self.create_rectangle(0, 0, width, 92, fill=BG_ALT, outline="")

        for idx in range(3):
            self.create_oval(-80 + (idx * 220), -40 + (idx * 8), 240 + (idx * 220), 190 + (idx * 12), fill="#132349", outline="", stipple="gray25")
        self.create_text(24, 20, anchor="nw", text="Bubble River", fill=TEXT, font=("Segoe UI", 18, "bold"))
        self.create_text(
            24,
            50,
            anchor="nw",
            text="Click for single-slot focus. Ctrl-click toggles. Shift-click spans a range. Drag a marquee to grab clusters.",
            fill=SUBTEXT,
            font=("Segoe UI", 9),
            width=max(240, width - 48),
        )

        if not self.controller.working_block_loaded():
            self.create_text(
                width / 2,
                height / 2,
                text="Load LINKFILE_000.bin or a saved unit block mod to light the bubble field.",
                fill=SUBTEXT,
                font=("Segoe UI", 12, "bold"),
            )
            return

        self.create_text(
            width - 22,
            22,
            anchor="ne",
            text=f"{UNIT_SLOT_COUNT} slots",
            fill=MUTED,
            font=("Consolas", 9),
        )

        start_row = max(0, int(max(0.0, self.scroll_y - (self.radius * 3)) / max(1.0, self.y_step)) - 2)
        end_row = min(rows - 1, int(max(0.0, self.scroll_y + height - self.pad_y + (self.radius * 3)) / max(1.0, self.y_step)) + 2)
        show_selected_labels = len(self.controller.selected_slot_set) <= 10

        for row in range(start_row, end_row + 1):
            base_index = row * self.columns
            for local_col in range(self.columns):
                slot_index = base_index + local_col
                if slot_index >= UNIT_SLOT_COUNT:
                    break
                x, y = self.slot_position(slot_index)
                palette = BUBBLE_PALETTE[self.bubble_palette_index(slot_index)]
                selected = slot_index in self.controller.selected_slot_set
                base_r = self.radius
                if selected:
                    halo = self.create_oval(x - base_r - 7, y - base_r - 7, x + base_r + 7, y + base_r + 7, outline=BUBBLE_GOLD, width=3)
                    glow = self.create_oval(
                        x - base_r - 12,
                        y - base_r - 12,
                        x + base_r + 12,
                        y + base_r + 12,
                        outline=BUBBLE_GOLD,
                        width=1,
                        stipple="gray25",
                    )
                    self.item_to_slot[halo] = slot_index
                    self.item_to_slot[glow] = slot_index
                shadow = self.create_oval(x - base_r + 3, y - base_r + 6, x + base_r + 3, y + base_r + 6, fill=SHADOW, outline="")
                self.item_to_slot[shadow] = slot_index
                orb = self.create_oval(
                    x - base_r,
                    y - base_r,
                    x + base_r,
                    y + base_r,
                    fill=palette["fill"],
                    outline=palette["outline"] if selected else "",
                    width=2 if selected else 0,
                )
                gloss = self.create_oval(
                    x - (base_r * 0.66),
                    y - (base_r * 0.80),
                    x - (base_r * 0.08),
                    y - (base_r * 0.22),
                    fill=palette["shine"],
                    outline="",
                    stipple="gray50",
                )
                glint = self.create_oval(
                    x - (base_r * 0.32),
                    y - (base_r * 0.54),
                    x - (base_r * 0.14),
                    y - (base_r * 0.36),
                    fill="#FFFFFF",
                    outline="",
                )
                self.item_to_slot[orb] = slot_index
                self.item_to_slot[gloss] = slot_index
                self.item_to_slot[glint] = slot_index
                if selected and show_selected_labels:
                    label_y = y - base_r - 17
                    pill = draw_round_rect(self, x - 28, label_y - 10, x + 28, label_y + 9, 10, fill=PANEL_ALT, outline=palette["outline"], width=1)
                    text = self.create_text(x, label_y - 1, text=str(slot_index), fill=TEXT, font=("Consolas", 9, "bold"))
                    self.item_to_slot[pill] = slot_index
                    self.item_to_slot[text] = slot_index

        if self.bubble_scrollmetrics(width, height, max_scroll):
            x1, y1, x2, y2 = self.scroll_bubble_bbox  # type: ignore[misc]
            self.create_oval(x1 + 3, y1 + 5, x2 + 3, y2 + 5, fill=SHADOW, outline="")
            self.create_oval(x1, y1, x2, y2, fill="#BF98D9", outline="#7D56A2", width=2)
            self.create_oval(
                x1 + (self.scroll_bubble_radius * 0.18),
                y1 + (self.scroll_bubble_radius * 0.16),
                x1 + (self.scroll_bubble_radius * 0.88),
                y1 + (self.scroll_bubble_radius * 0.84),
                fill="#EDD9FA",
                outline="",
                stipple="gray50",
            )
            self.create_oval(
                x1 + (self.scroll_bubble_radius * 0.56),
                y1 + (self.scroll_bubble_radius * 0.28),
                x1 + (self.scroll_bubble_radius * 0.82),
                y1 + (self.scroll_bubble_radius * 0.54),
                fill="#FFFFFF",
                outline="",
            )

        if self.dragging and self.drag_start and self.drag_current:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            draw_round_rect(self, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), 12, fill="#7DB4FF", outline="#B9D7FF", width=2, stipple="gray25")


class ModLibraryCanvas(tk.Canvas):
    def __init__(self, parent: tk.Misc, controller: "WO3UnitEditor"):
        super().__init__(parent, bg=PANEL, highlightthickness=0, bd=0, relief="flat")
        self.controller = controller
        self.files: List[str] = []
        self.item_to_path: Dict[int, str] = {}
        self.scroll_y = 0.0
        self.total_height = 0.0
        self.selected_path: Optional[str] = None
        self.bind("<Configure>", lambda _e: self.render())
        self.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<Button-1>", self.on_click)
        self.bind("<Double-Button-1>", self.on_double_click)

    def set_files(self, files: Sequence[str]) -> None:
        self.files = list(files)
        if self.selected_path not in self.files:
            self.selected_path = self.files[0] if self.files else None
        self.render()

    def on_mousewheel(self, event):
        if self.total_height <= self.winfo_height():
            return "break"
        delta = -1 if event.delta > 0 else 1
        max_scroll = max(0.0, self.total_height - max(1, self.winfo_height()))
        self.scroll_y = clamp(self.scroll_y + (delta * 64), 0.0, max_scroll)
        self.render()
        return "break"

    def path_at(self, x: float, y: float) -> Optional[str]:
        hits = self.find_overlapping(x, y, x, y)
        for item_id in reversed(hits):
            path = self.item_to_path.get(item_id)
            if path is not None:
                return path
        return None

    def on_click(self, event) -> None:
        path = self.path_at(event.x, event.y)
        if path:
            self.selected_path = path
            self.controller.update_mod_summary()
            self.render()

    def on_double_click(self, event) -> None:
        path = self.path_at(event.x, event.y)
        if path:
            self.selected_path = path
            self.controller.load_selected_library_mod()

    def render(self) -> None:
        self.delete("all")
        self.item_to_path.clear()
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self.create_rectangle(0, 0, width, height, fill=PANEL, outline="")
        if not self.files:
            self.create_text(
                width / 2,
                height / 2,
                text="No saved mod blocks yet.\nUse Save Mod As to create one here.",
                fill=SUBTEXT,
                font=("Segoe UI", 10),
                justify="center",
            )
            self.total_height = height
            self.scroll_y = 0.0
            return

        row_height = 78
        self.total_height = len(self.files) * row_height + 18
        max_scroll = max(0.0, self.total_height - height)
        self.scroll_y = clamp(self.scroll_y, 0.0, max_scroll)
        for index, path in enumerate(self.files):
            top = 10 + (index * row_height) - self.scroll_y
            bottom = top + row_height - 10
            if bottom < -6 or top > height + 6:
                continue
            selected = path == self.selected_path
            fill = "#243A71" if selected else PANEL_ALT
            outline = BUBBLE_GOLD if selected else PANEL_EDGE
            accent = BUBBLE_GOLD if selected else BUBBLE_BLUE
            draw_round_rect(self, 10, top, width - 12, bottom, 18, fill=fill, outline=outline, width=2)
            bubble = self.create_oval(22, top + 14, 58, top + 50, fill=accent, outline="")
            self.create_oval(28, top + 18, 42, top + 30, fill="#FFFFFF", outline="", stipple="gray50")
            name = os.path.basename(path)
            meta = f"{pretty_file_size(os.path.getsize(path))} | {datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')}"
            path_item = self.create_text(72, top + 16, anchor="nw", text=name, fill=TEXT, font=("Segoe UI", 10, "bold"))
            meta_item = self.create_text(72, top + 38, anchor="nw", text=meta, fill=SUBTEXT, font=("Segoe UI", 8))
            rel_item = self.create_text(72, top + 56, anchor="sw", text=safe_relpath(path), fill=MUTED, font=("Consolas", 8), width=max(160, width - 152))
            self.item_to_path[bubble] = path
            self.item_to_path[path_item] = path
            self.item_to_path[meta_item] = path
            self.item_to_path[rel_item] = path


def build_panel(parent: tk.Misc, title: str, subtitle: str, accent: str):
    shell = tk.Frame(parent, bg=BG)
    shell.grid_rowconfigure(1, weight=1)
    shell.grid_columnconfigure(0, weight=1)
    header = tk.Canvas(shell, height=102, bg=BG, highlightthickness=0, bd=0, relief="flat")
    body = tk.Frame(shell, bg=PANEL)
    header.grid(row=0, column=0, sticky="ew")
    body.grid(row=1, column=0, sticky="nsew")

    def draw_header(event) -> None:
        canvas = header
        width = max(1, event.width)
        height = max(1, event.height)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=BG_ALT, outline="")
        canvas.create_oval(-50, 8, 150, 150, fill=accent, outline="", stipple="gray25")
        canvas.create_oval(width - 180, -16, width + 18, 132, fill=accent, outline="", stipple="gray25")
        canvas.create_oval(width - 130, 12, width - 42, 88, fill=accent, outline="")
        canvas.create_oval(width - 112, 24, width - 84, 46, fill="#FFFFFF", outline="", stipple="gray50")
        for idx in range(5):
            y = 18 + (idx * 15)
            sway = 12 if idx % 2 else -12
            canvas.create_line(16, y, width - 16, y + sway, fill=HERO_LINE, width=1)
        canvas.create_text(20, 18, anchor="nw", text=title, fill=TEXT, font=("Segoe UI", 15, "bold"))
        canvas.create_text(20, 48, anchor="nw", text=subtitle, fill=SUBTEXT, font=("Segoe UI", 9), width=max(220, width - 44))

    header.bind("<Configure>", draw_header)
    return {"panel": shell, "body": body, "header": header}


class WO3UnitEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.title(APP_TITLE)
        self.configure(bg=BG)
        self.geometry("1740x1060")
        self.minsize(1480, 920)

        self.current_bin_path: Optional[str] = None
        self.current_mod_path: Optional[str] = None
        self.selected_library_path: Optional[str] = None
        self.mod_library_paths: List[str] = []
        self.current_source_label = "Memory buffer"
        self.backup_path: Optional[str] = None
        self.backup_block = b""
        self.compare_block = b""
        self.working_stream = io.BytesIO()
        self.working_origin_block = b""
        self.selected_slots: List[int] = []
        self.selected_slot_set = set()
        self.selection_anchor: Optional[int] = None
        self.changed_slots_vs_compare = set()
        self.changed_slots_vs_origin = set()
        self.field_editor_snapshot_slots: List[int] = []
        self.field_editor_snapshot_values: Dict[int, List[int]] = {}

        self.status_var = tk.StringVar(value="Open LINKFILE_000.bin or a saved unit mod to begin.")
        self.status_color = STATUS_GOOD
        self.selection_entry_var = tk.StringVar(value="")
        self.field_names = self.load_initial_field_names()

        self.field_cards: List[FieldCard] = []

        self.hero_canvas: Optional[tk.Canvas] = None
        self.toolbar_canvas: Optional[ToolbarCanvas] = None
        self.selection_summary_canvas: Optional[tk.Canvas] = None
        self.field_summary_canvas: Optional[tk.Canvas] = None
        self.mod_summary_canvas: Optional[tk.Canvas] = None
        self.editor_overlay: Optional[tk.Frame] = None
        self.editor_overlay_parent: Optional[tk.Misc] = None
        self.fields_canvas: Optional[tk.Canvas] = None
        self.fields_scrollbar: Optional[BubbleScrollbar] = None
        self.fields_window: Optional[int] = None
        self.fields_wrap: Optional[tk.Frame] = None
        self.bubble_canvas: Optional[BubbleGridCanvas] = None
        self.status_label: Optional[tk.Label] = None
        self.editor_panel_size = (840, 320)
        self.editor_panel_pos: Optional[Tuple[int, int]] = None
        self._editor_drag_offset: Optional[Tuple[int, int]] = None
        self.primary_slot: Optional[int] = None

        self.build_gui()
        self.refresh_mod_library()
        self.after(100, self.try_autoload_bin)

    def load_initial_field_names(self) -> List[str]:
        try:
            if os.path.isfile(FIELD_NAMES_JSON):
                return load_field_names_json(FIELD_NAMES_JSON)
        except Exception:
            pass
        return DEFAULT_FIELD_NAMES[:]

    def field_display_name(self, field_index: int) -> str:
        if 0 <= field_index < len(self.field_names):
            return self.field_names[field_index]
        return DEFAULT_FIELD_NAMES[field_index]

    def build_gui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.hero_canvas = tk.Canvas(self, height=112, bg=BG, highlightthickness=0, bd=0, relief="flat")
        self.hero_canvas.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        self.hero_canvas.bind("<Configure>", self.draw_hero)

        toolbar = tk.Frame(self, bg=BG)
        toolbar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        toolbar.grid_columnconfigure(0, weight=1)

        self.toolbar_canvas = ToolbarCanvas(toolbar, self)
        self.toolbar_canvas.grid(row=0, column=0, sticky="ew")
        self.mod_summary_canvas = None

        bubble_panel = build_panel(self, "Bubble River", "Each bubble is a unit's slot, select 1 or multiple and mod away.", BUBBLE_PINK)
        bubble_panel["panel"].grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 10))
        bubble_panel["header"].configure(height=76)
        bubble_panel["body"].grid_columnconfigure(0, weight=1)
        bubble_panel["body"].grid_rowconfigure(0, weight=1)

        self.bubble_canvas = BubbleGridCanvas(bubble_panel["body"], self)
        self.bubble_canvas.grid(row=0, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.editor_overlay_parent = bubble_panel["body"]

        self.editor_overlay = tk.Frame(self.editor_overlay_parent, bg=PANEL_ALT, highlightthickness=2, highlightbackground=PANEL_EDGE)
        self.editor_overlay.grid_columnconfigure(0, weight=1)
        self.editor_overlay.grid_rowconfigure(1, weight=1)

        self.selection_summary_canvas = tk.Canvas(self.editor_overlay, height=62, bg=PANEL_ALT, highlightthickness=0, bd=0, relief="flat", cursor="fleur")
        self.selection_summary_canvas.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 8))
        self.selection_summary_canvas.bind("<Configure>", lambda _e: self.update_selection_summary())
        self.selection_summary_canvas.bind("<ButtonPress-1>", self.start_editor_drag)
        self.selection_summary_canvas.bind("<B1-Motion>", self.drag_editor_panel)
        self.selection_summary_canvas.bind("<ButtonRelease-1>", self.end_editor_drag)
        self.field_summary_canvas = None

        self.fields_canvas = tk.Canvas(self.editor_overlay, bg=FIELD_DECK, highlightthickness=0, bd=0, relief="flat")
        self.fields_canvas.grid(row=1, column=0, sticky="nsew", padx=(14, 4))
        self.fields_canvas.bind("<MouseWheel>", self.on_fields_mousewheel)
        self.fields_scrollbar = BubbleScrollbar(self.editor_overlay, bubble_fill="#BF98D9")
        self.fields_scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 14))
        self.fields_scrollbar.set_command(self.on_fields_scrollbar_move)
        self.fields_canvas.configure(yscrollcommand=self.on_fields_canvas_scroll)
        self.fields_wrap = tk.Frame(self.fields_canvas, bg=FIELD_DECK)
        self.fields_wrap.grid_columnconfigure(0, weight=1)
        self.fields_wrap.grid_columnconfigure(1, weight=1)
        self.fields_wrap.grid_columnconfigure(2, weight=1)
        self.fields_wrap.grid_columnconfigure(3, weight=1)
        self.fields_window = self.fields_canvas.create_window((0, 0), window=self.fields_wrap, anchor="nw")
        self.fields_wrap.bind("<Configure>", lambda _e: self.refresh_fields_scrollregion())
        self.fields_canvas.bind("<Configure>", self.resize_fields_wrap)

        for index, field in enumerate(FIELD_SPECS):
            card = FieldCard(self.fields_wrap, self, index, field)
            row = index // 4
            column = index % 4
            pad_left = 0 if column == 0 else 4
            pad_right = 0 if column == 3 else 4
            card.grid(row=row, column=column, sticky="ew", padx=(pad_left, pad_right), pady=3)
            self.field_cards.append(card)

        overlay_actions = tk.Frame(self.editor_overlay, bg=PANEL_ALT)
        overlay_actions.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(8, 14))
        for col in range(6):
            overlay_actions.grid_columnconfigure(col, weight=1)
        GlowButton(overlay_actions, "Prev Slot", lambda: self.nudge_primary_slot(-1), fill=PANEL_ALT, outline=BUBBLE_BLUE, glow=BUBBLE_BLUE, height=36).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        GlowButton(overlay_actions, "Next Slot", lambda: self.nudge_primary_slot(1), fill=PANEL_ALT, outline=BUBBLE_BLUE, glow=BUBBLE_BLUE, height=36).grid(row=0, column=1, sticky="ew", padx=6)
        GlowButton(overlay_actions, "Apply To Selection", self.apply_fields_to_selection, fill=PANEL_ALT, outline=BUBBLE_GREEN, glow=BUBBLE_GREEN, height=36).grid(row=0, column=2, sticky="ew", padx=6)
        GlowButton(overlay_actions, "Reload Fields", self.reload_fields_from_snapshot, fill=PANEL_ALT, outline=BUBBLE_GOLD, glow=BUBBLE_GOLD, height=36).grid(row=0, column=3, sticky="ew", padx=6)
        GlowButton(overlay_actions, "Update Field Names", self.update_field_names, fill=PANEL_ALT, outline=BUBBLE_PURPLE, glow=BUBBLE_PURPLE, height=36).grid(row=0, column=4, sticky="ew", padx=6)
        GlowButton(overlay_actions, "Close Editor", self.clear_selection, fill=PANEL_ALT, outline=BUBBLE_PINK, glow=BUBBLE_PINK, height=36).grid(row=0, column=5, sticky="ew", padx=(6, 0))

        self.editor_overlay.place_forget()

        footer = tk.Frame(self, bg=PANEL_ALT, height=44)
        footer.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        footer.grid_propagate(False)
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = tk.Label(footer, textvariable=self.status_var, bg=PANEL_ALT, fg=self.status_color, anchor="w", font=("Segoe UI", 9, "bold"))
        self.status_label.grid(row=0, column=0, sticky="ew", padx=16, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_close_request)

    def build_input_card(self, parent: tk.Frame, *, row: int, column: int, variable: tk.StringVar, label: str) -> tk.Entry:
        card = tk.Canvas(parent, height=50, bg=parent.cget("bg"), highlightthickness=0, bd=0, relief="flat")
        card.grid(row=row, column=column, sticky="ew")
        entry = tk.Entry(card, textvariable=variable, relief="flat", bd=0, bg=FIELD_CARD, fg=FIELD_TEXT, insertbackground=FIELD_TEXT, font=("Consolas", 11))

        def redraw(event=None) -> None:
            width = max(120, card.winfo_width())
            height = max(50, card.winfo_height())
            card.delete("all")
            draw_round_rect(card, 0, 0, width - 2, height - 2, 18, fill=FIELD_CARD, outline=FIELD_BORDER, width=2)
            card.create_text(16, 14, anchor="nw", text=label, fill=FIELD_HINT, font=("Segoe UI", 8, "bold"))
            card.create_window(114, height / 2, anchor="w", width=max(160, width - 126), height=26, window=entry)

        card.bind("<Configure>", redraw)
        redraw()
        return entry

    def working_block_loaded(self) -> bool:
        return len(self.working_stream.getvalue()) == UNIT_BLOCK_SIZE

    def current_block(self) -> bytes:
        return self.working_stream.getvalue()

    def has_selection(self) -> bool:
        return bool(self.selected_slots)

    def set_status(self, text: str, color: str = STATUS_GOOD) -> None:
        self.status_var.set(text)
        self.status_color = color
        if self.status_label is not None:
            self.status_label.configure(fg=color)

    def draw_hero(self, event=None) -> None:
        if self.hero_canvas is None:
            return
        canvas = self.hero_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=BG, outline="")
        canvas.create_rectangle(0, 0, width, height, fill=BG_ALT, outline="")
        palette = [BUBBLE_PINK, BUBBLE_BLUE, BUBBLE_GOLD, "#FF503F", BUBBLE_GREEN, BUBBLE_PURPLE]
        bubble_x = 26
        bubble_y = 18
        radius = 20
        for index in range(12):
            row = index // 6
            col = index % 6
            x = bubble_x + (col * 40) + ((row % 2) * 20)
            y = bubble_y + (row * 34)
            color = palette[(row + col) % len(palette)]
            canvas.create_oval(x - radius + 2, y - radius + 3, x + radius + 2, y + radius + 3, fill=SHADOW, outline="")
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")
            canvas.create_oval(x - radius * 0.6, y - radius * 0.8, x - radius * 0.12, y - radius * 0.30, fill="#FFFFFF", outline="", stipple="gray50")
        for idx in range(4):
            y = 18 + (idx * 20)
            sway = 10 if idx % 2 else -8
            canvas.create_line(282, y, width - 20, y + sway, fill=HERO_LINE, width=1)
        canvas.create_text(300, 22, anchor="nw", text="WO3 Unit Data Editor", fill=TEXT, font=("Segoe UI", 22, "bold"))
        canvas.create_text(
            302,
            58,
            anchor="nw",
            text="Standalone Bubble Unit Data Editor.",
            fill=SUBTEXT,
            font=("Segoe UI", 9),
            width=max(260, width - 330),
        )

    def update_selection_summary(self) -> None:
        canvas = self.selection_summary_canvas
        if canvas is None:
            return
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=PANEL_ALT, outline="")
        draw_round_rect(canvas, 0, 0, width - 2, height - 2, 16, fill=PANEL, outline=PANEL_EDGE, width=1)
        canvas.create_text(width - 18, 10, anchor="ne", text="Drag", fill=MUTED, font=("Segoe UI", 8, "bold"))
        if not self.selected_slots:
            title = "Slot Editor"
            lines = [
                "No slots selected yet.",
                "Pick bubbles in the field, use a slot range, or drag a marquee.",
            ]
        else:
            title = f"Slot Editor | {len(self.selected_slots)} slot{'s' if len(self.selected_slots) != 1 else ''} selected"
        canvas.create_text(14, 10, anchor="nw", text=title, fill=TEXT, font=("Segoe UI", 12, "bold"))

    def update_field_summary(self) -> None:
        return

    def update_mod_summary(self) -> None:
        if self.toolbar_canvas is not None:
            self.toolbar_canvas.redraw()

    def start_editor_drag(self, event) -> None:
        if self.editor_overlay is None:
            return
        x = self.editor_overlay.winfo_x()
        y = self.editor_overlay.winfo_y()
        self._editor_drag_offset = (event.x, event.y)
        self.editor_panel_pos = (x, y)

    def drag_editor_panel(self, event) -> None:
        if self.editor_overlay is None or self.editor_overlay_parent is None or self._editor_drag_offset is None:
            return
        parent = self.editor_overlay_parent
        panel_w, panel_h = self.editor_panel_size
        rel_x = event.x_root - parent.winfo_rootx() - self._editor_drag_offset[0]
        rel_y = event.y_root - parent.winfo_rooty() - self._editor_drag_offset[1]
        max_x = max(0, parent.winfo_width() - panel_w - 8)
        max_y = max(0, parent.winfo_height() - panel_h - 8)
        self.editor_panel_pos = (
            int(clamp(rel_x, 8, max_x)),
            int(clamp(rel_y, 8, max_y)),
        )
        self.refresh_editor_overlay()

    def end_editor_drag(self, _event) -> None:
        self._editor_drag_offset = None

    def refresh_fields_scrollregion(self) -> None:
        if self.fields_canvas is None:
            return
        self.fields_canvas.configure(scrollregion=self.fields_canvas.bbox("all"))
        self.update_fields_scrollbar()

    def update_fields_scrollbar(self) -> None:
        if self.fields_canvas is None:
            return
        first, last = self.fields_canvas.yview()
        self.on_fields_canvas_scroll(first, last)

    def on_fields_canvas_scroll(self, first, last) -> None:
        if self.fields_canvas is None or self.fields_scrollbar is None:
            return
        viewport = max(1.0, self.fields_canvas.winfo_height())
        bbox = self.fields_canvas.bbox("all")
        content_height = max(viewport, float((bbox[3] - bbox[1]) if bbox else 1.0))
        span = min(1.0, viewport / max(1.0, content_height))
        max_first = max(0.0, 1.0 - span)
        raw_first = float(first)
        if max_first <= 0.0:
            self.fields_scrollbar.set(0.0, 1.0)
            return
        first_norm = raw_first / max_first
        self.fields_scrollbar.set(first_norm, min(1.0, first_norm + span))

    def on_fields_scrollbar_move(self, fraction: float) -> None:
        if self.fields_canvas is None:
            return
        viewport = max(1.0, self.fields_canvas.winfo_height())
        bbox = self.fields_canvas.bbox("all")
        content_height = max(viewport, float((bbox[3] - bbox[1]) if bbox else 1.0))
        span = min(1.0, viewport / max(1.0, content_height))
        max_first = max(0.0, 1.0 - span)
        self.fields_canvas.yview_moveto(0.0 if max_first <= 0.0 else clamp(fraction, 0.0, 1.0) * max_first)
        self.update_fields_scrollbar()

    def resize_fields_wrap(self, event) -> None:
        if self.fields_canvas is not None and self.fields_window is not None:
            self.fields_canvas.itemconfigure(self.fields_window, width=max(120, event.width))
            self.update_fields_scrollbar()

    def on_fields_mousewheel(self, event):
        if self.fields_canvas is None:
            return "break"
        delta = -1 if event.delta > 0 else 1
        self.fields_canvas.yview_scroll(delta * 3, "units")
        self.update_fields_scrollbar()
        return "break"

    def rebuild_diff_caches(self) -> None:
        current = self.current_block()
        self.changed_slots_vs_origin = set()
        self.changed_slots_vs_compare = set()
        if len(current) != UNIT_BLOCK_SIZE:
            return
        compare = self.compare_block if len(self.compare_block) == UNIT_BLOCK_SIZE else self.working_origin_block
        for slot_index in range(UNIT_SLOT_COUNT):
            start = slot_offset_in_block(slot_index)
            end = start + UNIT_SLOT_SIZE
            slot_bytes = current[start:end]
            if len(self.working_origin_block) == UNIT_BLOCK_SIZE and slot_bytes != self.working_origin_block[start:end]:
                self.changed_slots_vs_origin.add(slot_index)
            if len(compare) == UNIT_BLOCK_SIZE and slot_bytes != compare[start:end]:
                self.changed_slots_vs_compare.add(slot_index)

    def try_autoload_bin(self) -> None:
        candidates = find_candidate_bins()
        if candidates:
            self.load_bin(candidates[0], update_status=False)
            self.set_status(f"Auto-loaded {os.path.basename(candidates[0])}.", STATUS_GOOD)
        else:
            self.refresh_everything()

    def refresh_everything(self) -> None:
        self.refresh_editor_overlay()
        self.update_selection_summary()
        self.update_field_summary()
        self.update_mod_summary()
        if self.bubble_canvas is not None:
            self.bubble_canvas.render()

    def refresh_editor_overlay(self) -> None:
        if self.editor_overlay is None or self.editor_overlay_parent is None:
            return
        if self.working_block_loaded() and self.selected_slots:
            panel_w, panel_h = self.editor_panel_size
            max_x = max(8, self.editor_overlay_parent.winfo_width() - panel_w - 8)
            max_y = max(8, self.editor_overlay_parent.winfo_height() - panel_h - 8)
            if self.editor_panel_pos is None:
                self.editor_panel_pos = (max_x, 18)
            x = int(clamp(self.editor_panel_pos[0], 8, max_x))
            y = int(clamp(self.editor_panel_pos[1], 8, max_y))
            self.editor_panel_pos = (x, y)
            self.editor_overlay.place(x=x, y=y, width=panel_w, height=panel_h)
        else:
            self.editor_overlay.place_forget()

    def cycle_saved_mod(self, delta: int) -> None:
        paths = getattr(self, "mod_library_paths", [])
        if not paths:
            self.set_status("No saved mods are available in the mod folder yet.", STATUS_WARN)
            return
        if self.selected_library_path not in paths:
            self.selected_library_path = paths[0]
        else:
            index = paths.index(self.selected_library_path)
            self.selected_library_path = paths[(index + delta) % len(paths)]
        self.update_mod_summary()
        if self.selected_library_path:
            self.set_status(f"Selected saved mod {os.path.basename(self.selected_library_path)}.", STATUS_GOOD)

    def ensure_backup_for_bin(self, path: str, block: bytes) -> Tuple[bytes, str, bool]:
        target = backup_path_for_bin(path)
        created = False
        if os.path.isfile(target):
            backup = read_mod_block(target)
        else:
            write_mod_block(target, block)
            backup = block
            created = True
        return backup, target, created

    def confirm_discard_memory(self, reason: str) -> bool:
        if not self.changed_slots_vs_origin:
            return True
        return messagebox.askyesno(
            "Discard Unsaved Memory Changes?",
            f"Current in-memory changes have not been saved as a mod file.\n\nDiscard them before {reason}?",
        )

    def load_block_into_memory(
        self,
        block: bytes,
        *,
        source_label: str,
        source_path: Optional[str] = None,
        compare_block: Optional[bytes] = None,
        clear_selection: bool = True,
        update_status: bool = True,
    ) -> None:
        if len(block) != UNIT_BLOCK_SIZE:
            raise ValueError(f"Expected a {UNIT_BLOCK_SIZE} byte block, got {len(block)} bytes.")
        self.working_stream = io.BytesIO(block)
        self.working_origin_block = bytes(block)
        self.current_source_label = source_label
        self.current_mod_path = source_path
        if compare_block is not None:
            self.compare_block = compare_block
        elif len(self.backup_block) == UNIT_BLOCK_SIZE:
            self.compare_block = self.backup_block
        else:
            self.compare_block = self.working_origin_block
        self.rebuild_diff_caches()
        if clear_selection:
            self.set_selected_slots([], preserve_anchor=False)
        else:
            self.reload_fields_from_selection(silent=True)
        self.refresh_everything()
        if update_status:
            self.set_status(f"Loaded {source_label} into memory.", STATUS_GOOD)

    def open_bin_file(self) -> None:
        initial_dir = os.path.dirname(self.current_bin_path) if self.current_bin_path else SCRIPT_DIR
        path = filedialog.askopenfilename(
            title="Open LINKFILE_000.bin",
            initialdir=initial_dir,
            filetypes=[("BIN files", "*.bin *.BIN"), ("All files", "*.*")],
        )
        if path:
            self.load_bin(path)

    def load_bin(self, path: str, *, update_status: bool = True) -> None:
        if not self.confirm_discard_memory(f"loading {os.path.basename(path)}"):
            return
        try:
            block = read_unit_block_from_bin(path)
            backup, backup_path, created = self.ensure_backup_for_bin(path, block)
        except Exception as exc:
            messagebox.showerror("Open BIN Failed", str(exc))
            self.set_status(f"Could not load {os.path.basename(path)}.", STATUS_BAD)
            return
        self.current_bin_path = path
        self.backup_block = backup
        self.backup_path = backup_path
        self.load_block_into_memory(
            block,
            source_label=f"BIN | {os.path.basename(path)}",
            source_path=None,
            compare_block=backup,
            clear_selection=True,
            update_status=False,
        )
        if update_status:
            suffix = " Backup captured." if created else " Backup already available."
            self.set_status(f"Loaded {os.path.basename(path)} into memory.{suffix}", STATUS_GOOD)
        self.refresh_mod_library()

    def open_mod_file(self) -> None:
        if not self.confirm_discard_memory("loading a mod file"):
            return
        initial_dir = os.path.dirname(self.current_mod_path) if self.current_mod_path else MOD_DIR
        path = filedialog.askopenfilename(
            title="Open Saved WO3 Unit Block",
            initialdir=initial_dir,
            filetypes=[("WO3 unit block", "*.bin *.BIN"), ("All files", "*.*")],
        )
        if path:
            self.load_mod_path(path)

    def load_mod_path(self, path: str) -> None:
        try:
            block = read_mod_block(path)
        except Exception as exc:
            messagebox.showerror("Open Mod Failed", str(exc))
            self.set_status(f"Could not load {os.path.basename(path)}.", STATUS_BAD)
            return
        compare = self.backup_block if len(self.backup_block) == UNIT_BLOCK_SIZE else block
        self.load_block_into_memory(
            block,
            source_label=f"MOD | {os.path.basename(path)}",
            source_path=path,
            compare_block=compare,
            clear_selection=True,
            update_status=True,
        )
        self.selected_library_path = path
        self.update_mod_summary()

    def save_current_mod_as(self) -> None:
        if not self.working_block_loaded():
            messagebox.showinfo("Nothing To Save", "Load a BIN or a mod file first.")
            self.set_status("Nothing to save yet.", STATUS_WARN)
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_name = os.path.basename(self.current_mod_path) if self.current_mod_path else f"wo3_unit_mod_{stamp}.bin"
        path = filedialog.asksaveasfilename(
            title="Save Current Unit Block As",
            initialdir=MOD_DIR,
            initialfile=initial_name,
            defaultextension=".bin",
            filetypes=[("WO3 unit block", "*.bin"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            write_mod_block(path, self.current_block())
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))
            self.set_status(f"Could not save {os.path.basename(path)}.", STATUS_BAD)
            return
        self.current_mod_path = path
        self.current_source_label = f"MOD | {os.path.basename(path)}"
        self.working_origin_block = self.current_block()
        self.rebuild_diff_caches()
        self.refresh_mod_library(select_path=path)
        self.reload_fields_from_selection(silent=True)
        self.set_status(f"Saved {os.path.basename(path)}.", STATUS_GOOD)

    def apply_current_to_bin(self) -> None:
        if not self.current_bin_path:
            messagebox.showinfo("No BIN Loaded", "Open LINKFILE_000.bin first so the editor knows where to write the unit block.")
            self.set_status("Open a BIN before applying mods.", STATUS_WARN)
            return
        if not self.working_block_loaded():
            messagebox.showinfo("Nothing To Apply", "Load or create a unit block in memory first.")
            self.set_status("Nothing is loaded into memory yet.", STATUS_WARN)
            return
        try:
            write_unit_block_to_bin(self.current_bin_path, self.current_block())
        except Exception as exc:
            messagebox.showerror("Apply Failed", str(exc))
            self.set_status(f"Could not write to {os.path.basename(self.current_bin_path)}.", STATUS_BAD)
            return
        self.set_status(f"Applied current memory to {os.path.basename(self.current_bin_path)}.", STATUS_GOOD)

    def apply_selected_library_mod_to_bin(self) -> None:
        if not self.current_bin_path:
            messagebox.showinfo("No BIN Loaded", "Open LINKFILE_000.bin first so the editor knows where to write the unit block.")
            self.set_status("Open a BIN before applying a saved mod.", STATUS_WARN)
            return
        if not self.selected_library_path:
            messagebox.showinfo("No Saved Mod Selected", "Pick a saved mod block from the library first.")
            self.set_status("Choose a saved mod block first.", STATUS_WARN)
            return
        path = self.selected_library_path
        try:
            block = read_mod_block(path)
            write_unit_block_to_bin(self.current_bin_path, block)
        except Exception as exc:
            messagebox.showerror("Apply Selected Mod Failed", str(exc))
            self.set_status(f"Could not apply {os.path.basename(path)}.", STATUS_BAD)
            return
        self.set_status(f"Applied saved mod {os.path.basename(path)} to {os.path.basename(self.current_bin_path)}.", STATUS_GOOD)

    def restore_bin_from_backup(self) -> None:
        if not self.current_bin_path or not self.backup_path or len(self.backup_block) != UNIT_BLOCK_SIZE:
            messagebox.showinfo("No Backup Ready", "Open LINKFILE_000.bin first so the editor can create or locate the unit backup block.")
            self.set_status("A BIN backup is needed before restore can run.", STATUS_WARN)
            return
        if not messagebox.askyesno(
            "Restore Original Unit Block?",
            f"This will write the backed-up unit block back into:\n{self.current_bin_path}\n\nContinue?",
        ):
            return
        try:
            write_unit_block_to_bin(self.current_bin_path, self.backup_block)
        except Exception as exc:
            messagebox.showerror("Restore Failed", str(exc))
            self.set_status(f"Could not restore {os.path.basename(self.current_bin_path)}.", STATUS_BAD)
            return
        self.set_status(f"Restored the unit block in {os.path.basename(self.current_bin_path)} from backup.", STATUS_GOOD)

    def refresh_memory_from_source(self) -> None:
        if self.current_bin_path:
            try:
                block = read_unit_block_from_bin(self.current_bin_path)
                backup, backup_path, _created = self.ensure_backup_for_bin(self.current_bin_path, block)
            except Exception as exc:
                messagebox.showerror("Refresh Failed", str(exc))
                self.set_status(f"Could not refresh from {os.path.basename(self.current_bin_path)}.", STATUS_BAD)
                return
            self.backup_block = backup
            self.backup_path = backup_path
            self.load_block_into_memory(
                block,
                source_label=f"BIN | {os.path.basename(self.current_bin_path)}",
                source_path=None,
                compare_block=backup,
                clear_selection=True,
                update_status=False,
            )
            self.set_status(f"Refreshed memory from {os.path.basename(self.current_bin_path)}.", STATUS_GOOD)
            self.refresh_mod_library()
            return
        if self.current_mod_path:
            try:
                block = read_mod_block(self.current_mod_path)
            except Exception as exc:
                messagebox.showerror("Refresh Failed", str(exc))
                self.set_status(f"Could not refresh from {os.path.basename(self.current_mod_path)}.", STATUS_BAD)
                return
            compare = self.backup_block if len(self.backup_block) == UNIT_BLOCK_SIZE else block
            self.load_block_into_memory(
                block,
                source_label=f"MOD | {os.path.basename(self.current_mod_path)}",
                source_path=self.current_mod_path,
                compare_block=compare,
                clear_selection=True,
                update_status=False,
            )
            self.set_status(f"Refreshed memory from {os.path.basename(self.current_mod_path)}.", STATUS_GOOD)
            return
        messagebox.showinfo("Nothing To Refresh", "Open a BIN or load a mod file first.")
        self.set_status("Open a BIN or a mod file before refreshing memory.", STATUS_WARN)

    def refresh_mod_library(self, *, select_path: Optional[str] = None) -> None:
        os.makedirs(MOD_DIR, exist_ok=True)
        files = []
        for name in os.listdir(MOD_DIR):
            path = os.path.join(MOD_DIR, name)
            if not os.path.isfile(path):
                continue
            try:
                if os.path.getsize(path) == UNIT_BLOCK_SIZE:
                    files.append(path)
            except OSError:
                continue
        files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        self.mod_library_paths = files
        if select_path and select_path in files:
            self.selected_library_path = select_path
        elif self.selected_library_path not in files:
            self.selected_library_path = files[0] if files else None
        self.update_mod_summary()

    def load_selected_library_mod(self) -> None:
        if not self.selected_library_path:
            messagebox.showinfo("No Saved Mod Selected", "Pick a saved mod block from the library first.")
            self.set_status("Choose a saved mod block first.", STATUS_WARN)
            return
        if not self.confirm_discard_memory(f"loading {os.path.basename(self.selected_library_path)}"):
            return
        self.load_mod_path(self.selected_library_path)

    def slot_values_for_selection(self) -> Optional[List[List[int]]]:
        if not self.selected_slots or not self.working_block_loaded():
            return None
        block = self.current_block()
        return [read_slot_values(block, slot_index) for slot_index in self.selected_slots]

    def capture_field_editor_snapshot(self) -> None:
        if not self.selected_slots or not self.working_block_loaded():
            self.field_editor_snapshot_slots = []
            self.field_editor_snapshot_values = {}
            return
        snapshot_slots = self.selected_slots[:]
        block = self.current_block()
        self.field_editor_snapshot_slots = snapshot_slots
        self.field_editor_snapshot_values = {
            slot_index: read_slot_values(block, slot_index)
            for slot_index in snapshot_slots
        }

    def reload_fields_from_selection(self, *, silent: bool = False) -> None:
        values = self.slot_values_for_selection()
        if values is None:
            for card in self.field_cards:
                card.set_display("", mixed=False)
                card.set_enabled(False)
            self.update_field_summary()
            if not silent:
                self.set_status("No slots selected.", STATUS_WARN)
            return
        for index, field in enumerate(FIELD_SPECS):
            field_values = [slot_values[index] for slot_values in values]
            if all(value == field_values[0] for value in field_values[1:]):
                self.field_cards[index].set_display(format_field_value(field, field_values[0]), mixed=False)
            else:
                self.field_cards[index].set_display(MIXED_VALUE_TEXT, mixed=True)
        self.update_field_summary()

    def reload_fields_from_snapshot(self) -> None:
        if not self.selected_slots or not self.working_block_loaded():
            self.reload_fields_from_selection(silent=False)
            return
        if self.field_editor_snapshot_slots != self.selected_slots or not self.field_editor_snapshot_values:
            self.set_status("No field snapshot is ready for this selection yet.", STATUS_WARN)
            return
        new_block = bytearray(self.current_block())
        restored = 0
        for slot_index in self.selected_slots:
            values = self.field_editor_snapshot_values.get(slot_index)
            if not values or len(values) != len(FIELD_SPECS):
                continue
            restored += 1
            for field_index, field in enumerate(FIELD_SPECS):
                start = slot_offset_in_block(slot_index) + field.offset
                new_block[start : start + field.size] = values[field_index].to_bytes(field.size, "little", signed=False)
        if restored == 0:
            self.set_status("No field snapshot values were available to restore.", STATUS_WARN)
            return
        self.working_stream = io.BytesIO(bytes(new_block))
        self.rebuild_diff_caches()
        self.reload_fields_from_selection(silent=True)
        self.refresh_everything()
        self.set_status(f"Reloaded the original field snapshot for {restored} selected slot{'s' if restored != 1 else ''}.", STATUS_GOOD)

    def on_field_card_changed(self, _card: FieldCard) -> None:
        self.update_field_summary()

    def update_field_names(self) -> None:
        if not self.field_cards:
            return
        self.field_names = normalize_field_names([card.current_field_name() for card in self.field_cards])
        for index, card in enumerate(self.field_cards):
            card.set_field_name(self.field_names[index])
        try:
            save_field_names_json(FIELD_NAMES_JSON, self.field_names)
        except Exception as exc:
            messagebox.showerror("Field Name Save Failed", str(exc))
            self.set_status(f"Could not save field names: {exc}", STATUS_BAD)
            return
        self.set_status("Updated field names for the bubble editor.", STATUS_GOOD)

    def apply_fields_to_selection(self) -> None:
        if not self.selected_slots:
            messagebox.showinfo("No Selection", "Select one or more bubbles first.")
            self.set_status("Pick at least one slot before applying fields.", STATUS_WARN)
            return
        if not self.working_block_loaded():
            messagebox.showinfo("No Data Loaded", "Load a BIN or mod block first.")
            self.set_status("Load a BIN or a mod file first.", STATUS_WARN)
            return
        new_block = bytearray(self.current_block())
        for index, card in enumerate(self.field_cards):
            try:
                value = card.current_value()
            except ValueError as exc:
                card.entry.focus_set()
                try:
                    card.entry.selection_range(0, "end")
                except tk.TclError:
                    pass
                name = self.field_display_name(index)
                messagebox.showerror("Invalid Field Value", f"{name}: {exc}")
                self.set_status(f"{name} is invalid.", STATUS_BAD)
                return
            if value is None:
                continue
            field = FIELD_SPECS[index]
            raw_bytes = value.to_bytes(field.size, "little", signed=False)
            for slot_index in self.selected_slots:
                start = slot_offset_in_block(slot_index) + field.offset
                new_block[start : start + field.size] = raw_bytes
        self.working_stream = io.BytesIO(bytes(new_block))
        self.rebuild_diff_caches()
        self.reload_fields_from_selection(silent=True)
        self.refresh_everything()
        self.set_status(f"Applied field changes to {len(self.selected_slots)} selected slot{'s' if len(self.selected_slots) != 1 else ''}.", STATUS_GOOD)

    def set_selected_slots(self, slots: Sequence[int], *, preserve_anchor: bool = False, primary_slot: Optional[int] = None) -> None:
        valid = sorted({slot for slot in slots if 0 <= slot < UNIT_SLOT_COUNT})
        selection_changed = valid != self.selected_slots
        self.selected_slots = valid
        self.selected_slot_set = set(valid)
        if valid and not preserve_anchor:
            active = primary_slot if primary_slot in self.selected_slot_set else valid[-1]
            self.selection_anchor = active
            self.primary_slot = active
        elif valid and preserve_anchor:
            if self.primary_slot not in self.selected_slot_set:
                self.primary_slot = valid[-1]
            if self.selection_anchor not in self.selected_slot_set:
                self.selection_anchor = self.primary_slot
        elif not valid:
            self.selection_anchor = None
            self.primary_slot = None
        if selection_changed:
            self.capture_field_editor_snapshot()
        self.reload_fields_from_selection(silent=True)
        self.refresh_everything()

    def merge_selected_slots(self, slots: Sequence[int], *, primary_slot: Optional[int] = None) -> None:
        merged = sorted(self.selected_slot_set.union(slot for slot in slots if 0 <= slot < UNIT_SLOT_COUNT))
        self.set_selected_slots(merged, preserve_anchor=False, primary_slot=primary_slot)

    def toggle_slot(self, slot: int) -> None:
        if slot in self.selected_slot_set:
            new_slots = [value for value in self.selected_slots if value != slot]
            new_primary = self.primary_slot if self.primary_slot in new_slots else (new_slots[-1] if new_slots else None)
        else:
            new_slots = self.selected_slots + [slot]
            new_primary = slot
        self.set_selected_slots(new_slots, preserve_anchor=False, primary_slot=new_primary)

    def handle_bubble_click(self, slot: Optional[int], *, modifiers: int) -> None:
        if slot is None:
            if not (modifiers & (CTRL_MASK | SHIFT_MASK)):
                self.set_selected_slots([])
            return
        if modifiers & SHIFT_MASK and self.selection_anchor is not None:
            start = min(self.selection_anchor, slot)
            end = max(self.selection_anchor, slot)
            self.set_selected_slots(range(start, end + 1), preserve_anchor=False, primary_slot=slot)
        elif modifiers & CTRL_MASK:
            self.toggle_slot(slot)
        else:
            self.set_selected_slots([slot], preserve_anchor=False, primary_slot=slot)

    def clear_selection(self) -> None:
        self.set_selected_slots([])
        self.set_status("Selection cleared.", STATUS_GOOD)

    def invert_selection(self) -> None:
        inverted = [slot for slot in range(UNIT_SLOT_COUNT) if slot not in self.selected_slot_set]
        self.set_selected_slots(inverted)
        self.set_status("Selection inverted.", STATUS_GOOD)

    def select_changed_slots(self) -> None:
        if not self.changed_slots_vs_compare:
            self.set_selected_slots([])
            self.set_status("No slots differ from the base compare block yet.", STATUS_WARN)
            return
        self.set_selected_slots(sorted(self.changed_slots_vs_compare))
        self.set_status(f"Selected {len(self.changed_slots_vs_compare)} changed slot{'s' if len(self.changed_slots_vs_compare) != 1 else ''}.", STATUS_GOOD)

    def select_range_from_entry(self) -> None:
        try:
            slots = parse_slot_expression(self.selection_entry_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid Slot Expression", str(exc))
            self.set_status("Slot expression could not be parsed.", STATUS_BAD)
            return
        self.set_selected_slots(slots, primary_slot=slots[-1])
        self.set_status(f"Selected {len(slots)} slot{'s' if len(slots) != 1 else ''} from the range box.", STATUS_GOOD)

    def keep_last_slot_only(self) -> None:
        if not self.selected_slots:
            self.set_status("No selected slot to keep.", STATUS_WARN)
            return
        keep = self.primary_slot if self.primary_slot is not None else self.selected_slots[-1]
        self.set_selected_slots([keep], primary_slot=keep)
        self.set_status(f"Kept slot {keep} as the active selection.", STATUS_GOOD)

    def nudge_primary_slot(self, delta: int) -> None:
        if not self.selected_slots:
            target = 0 if delta <= 0 else UNIT_SLOT_COUNT - 1
        else:
            base = self.primary_slot if self.primary_slot is not None else self.selected_slots[-1]
            target = int(clamp(base + delta, 0, UNIT_SLOT_COUNT - 1))
        self.set_selected_slots([target], primary_slot=target)
        self.set_status(f"Selected slot {target}.", STATUS_GOOD)

    def on_close_request(self) -> None:
        if self.changed_slots_vs_origin and not messagebox.askyesno(
            "Close Editor?",
            "Current in-memory changes are not saved as a mod file.\n\nClose the editor anyway?",
        ):
            return
        self.destroy()


def report_fatal_error(exc: BaseException) -> None:
    ensure_app_dirs()
    trace = "".join(traceback.format_exception(exc))
    try:
        with open(CRASH_LOG_PATH, "w", encoding="utf-8") as handle:
            handle.write(trace)
    except OSError:
        pass
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        APP_TITLE,
        f"A fatal error occurred.\n\nA crash log was written to:\n{CRASH_LOG_PATH}\n\n{exc}",
    )
    root.destroy()


def main() -> None:
    app = WO3UnitEditor()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report_fatal_error(exc)
