from __future__ import annotations

import hashlib, io, json, os, sys, traceback
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence, Tuple

"""A standalone Unit Data Editor for WO3, Steel themed to match Tetsutetsu from mha"""

SCRIPT_PATH = os.path.abspath(sys.argv[0] or __file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
os.chdir(SCRIPT_DIR)

APP_TITLE = "WO3 Steel Unit Editor"
BIN_FILENAME = "LINKFILE_000.bin"
BIN_FILENAME_UPPER = "LINKFILE_000.BIN"
UNIT_DATA_OFFSET = 0x44CDA
UNIT_SLOT_COUNT = 2206
UNIT_SLOT_SIZE = 40
UNIT_BLOCK_SIZE = UNIT_SLOT_COUNT * UNIT_SLOT_SIZE
FIELD_NAME_LIMIT = 40
MAX_DRAWN_ROWS = 100

APP_DATA_DIR = os.path.join(SCRIPT_DIR, "wo3_editor")
MOD_DIR = os.path.join(APP_DATA_DIR, "mods")
BACKUP_DIR = os.path.join(APP_DATA_DIR, "backups")
FIELD_NAMES_JSON = os.path.join(APP_DATA_DIR, "field_names.json")
CRASH_LOG_PATH = os.path.join(APP_DATA_DIR, "wo3_editor_crash.log")

WINDOW_BG = "#5E6976"
PANEL_BG = "#6F7C89"
PANEL_ALT = "#7C8996"
PANEL_EDGE = "#D6DEE5"
HEADER_BG = "#56616E"
GRID_BG = "#EEF2F5"
GRID_ALT = "#E4EAF0"
GRID_LINE = "#C7D0D7"
GRID_TEXT = "#243240"
GRID_MUTED = "#617485"
SELECT_FILL = "#D9E6F6"
SELECT_EDGE = "#7890AA"
BUTTON_FILL = "#707D8A"
BUTTON_FILL_HOVER = "#7D8A98"
BUTTON_EDGE = "#E2E8ED"
BUTTON_TEXT = "#F4F7FA"
STATUS_BG = "#4D5865"
STATUS_TEXT = "#F7FAFC"
ACCENT = "#B9C5D1"
INPUT_BG = "#F8FBFD"
INPUT_TEXT = "#22303E"
INPUT_EDGE = "#A8B6C4"
GOOD = "#B9EBC8"
WARN = "#F6E1A1"
BAD = "#F4B7BE"
SHADOW = "#39434E"
PRIMARY_SELECT_FILL = "#C7D8ED"
PRIMARY_INDEX_FILL = "#C4D6EA"
MIXED_VALUE_TEXT = "Mixed Value"

CTRL_MASK = 0x0004
SHIFT_MASK = 0x0001

class Tooltip:
    def __init__(self, widget):
        self.widget = widget
        self.tip = None

    def show(self, text, x, y):
        if not text:
            return
        self.hide()

        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.geometry(f"+{x+12}+{y+12}")

        label = tk.Label(self.tip, text=text, bg="#222", fg="white",
                         font=("Segoe UI", 9), padx=6, pady=3)
        label.pack()

    def hide(self):
        if self.tip:
            self.tip.destroy()
            self.tip = None

@dataclass(frozen=True)
class FieldSpec:
    name: str
    size: int
    offset: int


def load_unit_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def load_id_name_map(path: str) -> Dict[int, str]:
    mapping = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            try:
                left, right = line.split(":", 1)
                key = int(left.strip())
                value = right.strip()
                mapping[key] = value
            except:
                continue

    return mapping

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


def rounded_rect_points(x1: float, y1: float, x2: float, y2: float, radius: float) -> List[float]:
    radius = max(1.0, min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    return [
        x1 + radius, y1,
        x1 + radius, y1,
        x2 - radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1 + radius,
        x1, y1,
        x1 + radius, y1,
    ]


def draw_round_rect(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, radius: float, **kwargs):
    return canvas.create_polygon(rounded_rect_points(x1, y1, x2, y2, radius), smooth=True, splinesteps=36, **kwargs)


def int_bits_for_size(size: int) -> int:
    return size * 8


def mask_for_size(size: int) -> int:
    return (1 << int_bits_for_size(size)) - 1


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


def slot_offset_in_block(slot_index: int) -> int:
    return slot_index * UNIT_SLOT_SIZE


def read_unit_block_from_bin(path: str) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        end = UNIT_DATA_OFFSET + UNIT_BLOCK_SIZE
        if size < end:
            raise ValueError(f"{os.path.basename(path)} is only {size} bytes, but the unit block ends at 0x{end:X}.")
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
            raise ValueError(f"{os.path.basename(path)} is only {size} bytes, but the unit block ends at 0x{end:X}.")
        handle.seek(UNIT_DATA_OFFSET)
        handle.write(block)


def read_mod_block(path: str) -> bytes:
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) != UNIT_BLOCK_SIZE:
        raise ValueError(f"{os.path.basename(path)} is {len(blob)} bytes, but a WO3 unit block mod must be {UNIT_BLOCK_SIZE} bytes.")
    return blob


def write_mod_block(path: str, block: bytes) -> None:
    if len(block) != UNIT_BLOCK_SIZE:
        raise ValueError(f"Expected a {UNIT_BLOCK_SIZE} byte block, got {len(block)} bytes.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(block)


def read_slot_values(block: bytes, slot_index: int) -> List[int]:
    base = slot_offset_in_block(slot_index)
    values: List[int] = []
    for field in FIELD_SPECS:
        start = base + field.offset
        values.append(int.from_bytes(block[start : start + field.size], "little", signed=False))
    return values


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


def summarize_ranges(slots: Sequence[int], *, limit: int = 6) -> str:
    if not slots:
        return "None"
    ordered = sorted(dict.fromkeys(slots))
    groups: List[str] = []
    start = prev = ordered[0]
    for slot in ordered[1:]:
        if slot == prev + 1:
            prev = slot
            continue
        groups.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = slot
    groups.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(groups[:limit]) + (" ..." if len(groups) > limit else "")


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


def normalize_field_names(names: Sequence[str]) -> List[str]:
    cleaned = [str(name).strip()[:FIELD_NAME_LIMIT] for name in names]
    if len(cleaned) != len(DEFAULT_FIELD_NAMES):
        cleaned = []
    result = []
    for index, default in enumerate(DEFAULT_FIELD_NAMES):
        result.append(cleaned[index] if index < len(cleaned) and cleaned[index] else default)
    return result


def save_field_names_json(path: str, names: Sequence[str], column_widths: Sequence[int]) -> None:
    payload = {
        "field_names": normalize_field_names(names),
        "column_widths": list(column_widths),
        "slot_size": UNIT_SLOT_SIZE,
        "slot_count": UNIT_SLOT_COUNT,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def load_field_names_json(path: str) -> Tuple[List[str], List[int]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        names = payload.get("field_names", [])
        widths = payload.get("column_widths", [])
    elif isinstance(payload, list):
        names = payload
        widths = []
    else:
        raise ValueError("Invalid field name JSON format.")

    return normalize_field_names(names), list(widths)


class SteelButton(tk.Canvas):
    def __init__(self, parent: tk.Misc, text: str, command, *, width: int = 112, height: int = 30):
        super().__init__(parent, width=width, height=height, bg=parent.cget("bg"), highlightthickness=0, bd=0, relief="flat", cursor="hand2")
        self._text = text
        self._command = command
        self._hover = False
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonRelease-1>", self.on_click)
        self.redraw()

    def on_enter(self, _event) -> None:
        self._hover = True
        self.redraw()

    def on_leave(self, _event) -> None:
        self._hover = False
        self.redraw()

    def on_click(self, _event) -> None:
        if callable(self._command):
            self._command()

    def redraw(self) -> None:
        self.delete("all")
        width = max(20, self.winfo_width())
        height = max(20, self.winfo_height())
        fill = BUTTON_FILL_HOVER if self._hover else BUTTON_FILL
        draw_round_rect(self, 1, 1, width - 2, height - 2, 10, fill=fill, outline=BUTTON_EDGE, width=1)
        self.create_text(width / 2, height / 2, text=self._text, fill=BUTTON_TEXT, font=("Segoe UI", 9, "bold"))


class SteelKnobScrollbar(tk.Canvas):
    def __init__(self, parent: tk.Misc, *, orient: str):
        if orient not in ("vertical", "horizontal"):
            raise ValueError("orient must be 'vertical' or 'horizontal'")
        width = 28 if orient == "vertical" else 180
        height = 180 if orient == "vertical" else 28
        super().__init__(parent, width=width, height=height, bg=parent.cget("bg"), highlightthickness=0, bd=0, relief="flat", cursor="hand2")
        self.orient = orient
        self._command = None
        self._first = 0.0
        self._last = 1.0
        self._dragging = False
        self._drag_offset = 0.0
        self._bubble_bbox: Optional[Tuple[float, float, float, float]] = None
        self._bubble_center = 0.0
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)

    def set_command(self, command) -> None:
        self._command = command

    def set_view(self, first: float, last: float) -> None:
        self._first = clamp(first, 0.0, 1.0)
        self._last = clamp(last, 0.0, 1.0)
        self.redraw()

    def metrics(self) -> Optional[Tuple[float, float, float, float]]:
        if self._last - self._first >= 0.999:
            self._bubble_bbox = None
            return None
        radius = 11.0
        if self.orient == "vertical":
            length = max(80, self.winfo_height())
            origin = 8.0
            travel = max(1.0, length - 16.0 - (radius * 2.0))
            center = origin + radius + (travel * self._first)
            x = self.winfo_width() / 2.0
            self._bubble_bbox = (x - radius, center - radius, x + radius, center + radius)
        else:
            length = max(120, self.winfo_width())
            origin = 8.0
            travel = max(1.0, length - 16.0 - (radius * 2.0))
            center = origin + radius + (travel * self._first)
            y = self.winfo_height() / 2.0
            self._bubble_bbox = (center - radius, y - radius, center + radius, y + radius)
        self._bubble_center = center
        return center, radius, travel, origin

    def move_to(self, center: float) -> None:
        if self._command is None:
            return
        metrics = self.metrics()
        if metrics is None:
            return
        _, radius, travel, origin = metrics
        min_center = origin + radius
        fraction = (clamp(center, min_center, min_center + travel) - min_center) / max(1.0, travel)
        self._command(fraction)

    def on_press(self, event) -> None:
        metrics = self.metrics()
        if metrics is None or self._bubble_bbox is None:
            return
        x1, y1, x2, y2 = self._bubble_bbox
        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            self._dragging = True
            coord = event.y if self.orient == "vertical" else event.x
            self._drag_offset = coord - self._bubble_center

    def on_drag(self, event) -> None:
        if not self._dragging:
            return
        coord = event.y if self.orient == "vertical" else event.x
        self.move_to(coord - self._drag_offset)

    def on_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        coord = event.y if self.orient == "vertical" else event.x
        offset = self._drag_offset
        self._drag_offset = 0.0
        self.move_to(coord - offset)

    def redraw(self) -> None:
        self.delete("all")
        metrics = self.metrics()
        if metrics is None or self._bubble_bbox is None:
            return
        x1, y1, x2, y2 = self._bubble_bbox
        self.create_oval(x1 + 2, y1 + 3, x2 + 2, y2 + 3, fill=SHADOW, outline="")
        self.create_oval(x1, y1, x2, y2, fill="#BBC6D0", outline="#647283", width=2)
        self.create_oval(x1 + 3, y1 + 3, x2 - 4, y2 - 4, fill="#E9EFF4", outline="", stipple="gray50")
        self.create_oval(x1 + 7, y1 + 4, x1 + 12, y1 + 9, fill="#FFFFFF", outline="")


class WO3SteelEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.title(APP_TITLE)
        self.geometry("1700x960")
        self.minsize(1260, 760)
        self.configure(bg=WINDOW_BG)

        self.tooltip = Tooltip(self)

        self.current_bin_path: Optional[str] = None
        self.current_mod_path: Optional[str] = None
        self.backup_path: Optional[str] = None
        self.backup_block = b""
        self.compare_block = b""
        self.working_origin_block = b""
        self.working_stream = io.BytesIO()
        self.slot_cache: List[List[int]] = []
        self.original_order: List[int] = list(range(UNIT_SLOT_COUNT))
        self.view_order: List[int] = self.original_order[:]
        self.view_lookup: Dict[int, int] = {slot: slot for slot in self.original_order}
        self.sort_field: Optional[int] = None
        self.sort_reverse = False
        self.selected_slots: List[int] = []
        self.selected_slot_set = set()
        self.selection_anchor: Optional[int] = None
        self.primary_slot: Optional[int] = None
        self.changed_slots_vs_origin = set()
        self.status_var = tk.StringVar(value="Open LINKFILE_000.bin or a saved unit mod to begin.")
        self.range_var = tk.StringVar(value="")
        self.field_names = self.load_initial_field_names()
        self.field_name_vars: List[tk.StringVar] = []
        self.field_name_entries: List[tk.Entry] = []
        self.row_height = 26
        self.header_height = 64
        self.index_width = 64
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.slot_sort_box: Optional[Tuple[float, float, float, float]] = None
        self.header_sort_boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
        self.inline_editor: Optional[tk.Entry] = None
        self.inline_edit_var = tk.StringVar()
        self.inline_edit_context: Optional[Tuple[int, int, bool]] = None
        self.unit_names: List[str] = []
        self.model_names = {}
        self.voice_names = {}
        self.moveset_names = {}

        model_path = os.path.join(APP_DATA_DIR, "WO3DE_Models.txt")
        voice_path = os.path.join(APP_DATA_DIR, "WO3DE_Voices.txt")
        moveset_path = os.path.join(APP_DATA_DIR, "WO3DE_Moveset.txt")
        names_path = os.path.join(APP_DATA_DIR, "wo3_names.txt")

        if os.path.isfile(model_path):
            self.model_names = load_id_name_map(model_path)

        if os.path.isfile(voice_path):
            self.voice_names = load_id_name_map(voice_path)

        if os.path.isfile(moveset_path):
            self.moveset_names = load_id_name_map(moveset_path)
        if os.path.isfile(names_path):
            self.unit_names = load_unit_names(names_path)

        self.title_canvas: Optional[tk.Canvas] = None
        self.header_canvas: Optional[tk.Canvas] = None
        self.body_canvas: Optional[tk.Canvas] = None
        self.v_scroll: Optional[SteelKnobScrollbar] = None
        self.h_scroll: Optional[SteelKnobScrollbar] = None
        self.status_label: Optional[tk.Label] = None
        self.header_background_id: Optional[int] = None
        self.header_index_ids: Dict[str, int] = {}
        self.header_column_items: List[Dict[str, int]] = []
        self.body_background_id: Optional[int] = None
        self.body_empty_text_id: Optional[int] = None
        self.body_row_items: List[Dict[str, object]] = []
        self.multi_editor_window: Optional[tk.Toplevel] = None
        self.multi_editor_title_var = tk.StringVar(value="")
        self.multi_editor_info_var = tk.StringVar(value="")
        self.multi_editor_canvas: Optional[tk.Canvas] = None
        self.multi_editor_inner: Optional[tk.Frame] = None
        self.multi_editor_window_id: Optional[int] = None
        self.multi_editor_scroll: Optional[SteelKnobScrollbar] = None
        self.multi_editor_vars: List[tk.StringVar] = []
        self.multi_editor_labels: List[tk.Label] = []
        self.multi_editor_entries: List[tk.Entry] = []
        self.multi_editor_mixed: List[bool] = []
        self.multi_editor_snapshot_slots: List[int] = []
        self.multi_editor_snapshot_values: Dict[int, List[int]] = {}

        self.build_gui()
        if os.path.isfile(FIELD_NAMES_JSON):
            self.set_status("Loaded saved field configuration.", GOOD)
        else:
            self.set_status("Using default field configuration.", STATUS_TEXT)
        self.after(100, self.try_autoload_bin)

    def on_body_hover(self, event):
        if not self.working_block_loaded():
            self.tooltip.hide()
            return

        row = int((event.y + self.y_offset) // self.row_height)
        if row < 0 or row >= len(self.view_order):
            self.tooltip.hide()
            return

        col_x = event.x + self.x_offset

        x = 0
        for field_index, width in enumerate(self.column_widths):
            if x <= col_x < x + width:
                slot = self.view_order[row]
                value = self.slot_cache[slot][field_index]
                field_name = FIELD_SPECS[field_index].name

                text = ""

                if field_name == "Voice ID":
                    text = self.voice_names.get(value, "")
                elif field_name == "Model ID":
                    text = self.model_names.get(value, "")
                elif field_name == "Moveset":
                    text = self.moveset_names.get(value, "")
                elif field_name == "Name":
                    if 0 <= value < len(self.unit_names):
                        text = self.unit_names[value]

                if text:
                    self.tooltip.show(text, event.x_root, event.y_root)
                else:
                    self.tooltip.hide()
                return

            x += width

        self.tooltip.hide()

    def clip_text_to_width(self, text, width_px):
        if not text:
            return ""

        avg_char_width = 7
        max_chars = max(3, (width_px - 12) // avg_char_width)

        if len(text) <= max_chars:
            return text

        return text[:max_chars - 3] + "..."

    def load_initial_field_names(self) -> List[str]:
        try:
            if os.path.isfile(FIELD_NAMES_JSON):
                names, widths = load_field_names_json(FIELD_NAMES_JSON)

                if widths and len(widths) == len(FIELD_SPECS):
                    self.column_widths = widths
                else:
                    self.column_widths = []

                return names
        except Exception:
            pass

        self.column_widths = []
        return DEFAULT_FIELD_NAMES[:]

    def on_frozen_hover(self, event):
        canvas = event.widget

        if not self.working_block_loaded():
            self.tooltip.hide()
            return

        row = int((event.y + self.y_offset) // self.row_height)
        if row < 0 or row >= len(self.view_order):
            self.tooltip.hide()
            return

        slot_index = self.view_order[row]
        values = self.slot_cache[slot_index]

        text = ""

        if canvas == self.body_name_canvas:
            name_id = values[0]
            if 0 <= name_id < len(self.unit_names):
                text = self.unit_names[name_id]

        elif canvas == self.body_model_canvas:
            text = self.model_names.get(values[4], "")

        elif canvas == self.body_voice_canvas:
            text = self.voice_names.get(values[3], "")

        elif canvas == self.body_moveset_canvas:
            text = self.moveset_names.get(values[5], "")

        if text:
            self.tooltip.show(text, event.x_root, event.y_root)
        else:
            self.tooltip.hide()

    def on_select_all(self, event=None):
        if not self.working_block_loaded():
            return "break"

        if len(self.selected_slot_set) == len(self.view_order):
            self.selected_slot_set.clear()
            self.selected_slots = []
            self.primary_slot = None
        else:
            self.selected_slot_set = set(self.view_order)
            self.selected_slots = list(self.view_order)
            self.primary_slot = self.view_order[0] if self.view_order else None

        self.refresh_multi_editor_from_selection()

        self.render_table()
        return "break"

    def apply_column_width(self):
        try:
            width = int(self.column_width_var.get())
        except ValueError:
            self.set_status("Invalid width value", BAD)
            return

        if width < 30:
            self.set_status("Width too small", BAD)
            return

        col_index = self.column_selector.current()

        if col_index < 0 or col_index >= len(self.column_widths):
            return

        self.column_widths[col_index] = width

        self.render_header()
        self.render_body()
        self.update_scrollbars()
        save_field_names_json(FIELD_NAMES_JSON, self.field_names, self.column_widths)

    def on_column_select(self, event=None):
        idx = self.column_selector.current()
        if 0 <= idx < len(self.column_widths):
            self.column_width_var.set(str(self.column_widths[idx]))

    def build_gui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.title_canvas = tk.Canvas(self, height=72, bg=WINDOW_BG, highlightthickness=0, bd=0, relief="flat")
        self.title_canvas.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        self.title_canvas.bind("<Configure>", self.draw_title)

        self.topbar = tk.Frame(self, bg=WINDOW_BG)
        self.topbar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        self.topbar_inner = tk.Frame(self.topbar, bg=WINDOW_BG)
        self.topbar_inner.pack(fill="x")

        self.topbar_widgets = []

        buttons = [
            ("Open BIN", self.open_bin_file),
            ("Load Mod", self.open_mod_file),
            ("Save Mod", self.save_current_mod_as),
            ("Apply BIN", self.apply_current_to_bin),
            ("Restore BIN", self.restore_bin_from_backup),
            ("Refresh", self.refresh_memory_from_source),
            ("Original", self.clear_sort),
            ("Export Names", self.export_field_names),
            ("Import Names", self.import_field_names),
        ]

        for text, command in buttons:
            btn = SteelButton(self.topbar_inner, text, command, width=116, height=30)
            self.topbar_widgets.append(btn)

        range_card = tk.Canvas(self.topbar_inner, width=212, height=30, bg=WINDOW_BG, highlightthickness=0, bd=0)
        range_entry = tk.Entry(range_card, textvariable=self.range_var, relief="flat", bd=0,
                               bg=INPUT_BG, fg=INPUT_TEXT, insertbackground=INPUT_TEXT,
                               font=("Consolas", 10))

        range_entry.bind("<Return>", lambda _e: self.select_range_from_entry())

        def redraw_range(_event=None):
            range_card.delete("all")
            width = max(120, range_card.winfo_width())
            height = max(24, range_card.winfo_height())

            draw_round_rect(range_card, 1, 1, width - 2, height - 2, 10,
                            fill=INPUT_BG, outline=INPUT_EDGE, width=1)

            range_card.create_text(10, height / 2, anchor="w",
                                   text="Slot / Range", fill=GRID_MUTED,
                                   font=("Segoe UI", 8, "bold"))

            range_card.create_window(width - 112, height / 2,
                                     anchor="w", width=102, height=20,
                                     window=range_entry)

        range_card.bind("<Configure>", redraw_range)
        redraw_range()

        self.topbar_widgets.append(range_card)

        go_btn = SteelButton(self.topbar_inner, "Go", self.select_range_from_entry, width=56, height=30)
        self.topbar_widgets.append(go_btn)

        self.column_selector = ttk.Combobox(self.topbar_inner, state="readonly", width=18)
        self.column_selector["values"] = [f.name for f in FIELD_SPECS]
        self.column_selector.current(0)
        self.topbar_widgets.append(self.column_selector)

        self.column_width_var = tk.StringVar(value="100")

        self.column_width_entry = tk.Entry(self.topbar_inner, textvariable=self.column_width_var, width=6)
        self.topbar_widgets.append(self.column_width_entry)

        set_width_btn = SteelButton(self.topbar_inner, "Set Width", self.apply_column_width, width=100, height=30)
        self.topbar_widgets.append(set_width_btn)

        self.column_selector.bind("<<ComboboxSelected>>", self.on_column_select)

        self.topbar.bind("<Configure>", lambda e: self.layout_topbar())

        table_frame = tk.Frame(self, bg=WINDOW_BG)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 10))
        table_frame.grid_columnconfigure(0, weight=0)
        table_frame.grid_columnconfigure(1, weight=0)
        table_frame.grid_columnconfigure(2, weight=0)
        table_frame.grid_columnconfigure(3, weight=0)
        table_frame.grid_columnconfigure(4, weight=0)
        table_frame.grid_columnconfigure(5, weight=1)
        table_frame.grid_columnconfigure(6, weight=0)
        table_frame.grid_rowconfigure(1, weight=1)
        self.header_index_canvas = tk.Canvas(
            table_frame,
            width=self.index_width,
            height=self.header_height,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=PANEL_EDGE,
            bd=0
        )
        self.header_index_canvas.grid(row=0, column=0, sticky="ns")
        self.header_canvas = tk.Canvas(
            table_frame,
            height=self.header_height,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=PANEL_EDGE,
            bd=0
        )
        self.header_canvas.grid(row=0, column=5, sticky="nsew")
        self.header_canvas.bind("<Configure>", lambda _e: self.render_table())
        self.header_canvas.bind("<Button-1>", self.on_header_click)
        self.body_index_canvas = tk.Canvas(
            table_frame,
            width=self.index_width,
            bg=GRID_BG,
            highlightthickness=1,
            highlightbackground=PANEL_EDGE,
            bd=0
        )
        self.body_index_canvas.grid(row=1, column=0, sticky="ns")
        self.name_width = 180
        self.model_width = 160
        self.voice_width = 160
        self.moveset_width = 160
        self.header_name_canvas = tk.Canvas(
            table_frame,
            width=self.name_width,
            height=self.header_height,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground=PANEL_EDGE,
            bd=0
        )
        self.header_name_canvas.grid(row=0, column=1, sticky="nsew")
        self.body_name_canvas = tk.Canvas(
            table_frame,
            width=self.name_width,
            bg=GRID_BG,
            highlightthickness=1,
            highlightbackground=PANEL_EDGE,
            bd=0
        )
        self.body_name_canvas.grid(row=1, column=1, sticky="nsew")
        self.body_canvas = tk.Canvas(
            table_frame,
            bg=GRID_BG,
            highlightthickness=1,
            highlightbackground=PANEL_EDGE,
            bd=0
        )
        
        self.body_canvas.grid(row=1, column=5, sticky="nsew")

        self.body_canvas.bind("<Configure>", lambda _e: self.render_table())
        self.body_canvas.bind("<MouseWheel>", self.on_body_mousewheel)
        self.body_canvas.bind("<Shift-MouseWheel>", self.on_body_shift_mousewheel)
        self.body_canvas.bind("<Button-1>", self.on_body_click)
        self.body_canvas.bind("<Double-Button-1>", self.on_body_double_click)
        self.body_canvas.bind("<Motion>", self.on_body_hover)
        self.body_canvas.bind("<Leave>", lambda e: self.tooltip.hide())
        self.bind_all("<Control-a>", self.on_select_all)

        self.v_scroll = SteelKnobScrollbar(table_frame, orient="vertical")
        self.v_scroll.grid(row=1, column=6, sticky="ns")
        self.v_scroll.set_command(self.on_vertical_scroll)

        self.h_scroll = SteelKnobScrollbar(table_frame, orient="horizontal")
        self.h_scroll.grid(row=2, column=0, columnspan=6, sticky="ew")
        self.h_scroll.set_command(self.on_horizontal_scroll)

        footer = tk.Frame(self, bg=STATUS_BG, height=38)
        footer.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))
        footer.grid_propagate(False)
        self.status_label = tk.Label(footer, textvariable=self.status_var, bg=STATUS_BG, fg=STATUS_TEXT, anchor="w", font=("Segoe UI", 9, "bold"))
        self.status_label.pack(fill="both", padx=14)

        self.prepare_header_entries()
        loaded_widths = self.column_widths[:]

        self.recompute_column_widths()

        if loaded_widths and len(loaded_widths) == len(FIELD_SPECS):
            self.column_widths = loaded_widths
        self.protocol("WM_DELETE_WINDOW", self.on_close_request)

    def layout_topbar(self):
        if not hasattr(self, "topbar_widgets"):
            return

        width = self.topbar.winfo_width()
        if width <= 1:
            return

        padding = 8
        x = 0
        y = 0
        row_height = 36

        for widget in self.topbar_widgets:
            widget.update_idletasks()
            w = widget.winfo_reqwidth()
            h = widget.winfo_reqheight()

            if x + w > width - 10:
                x = 0
                y += row_height

            widget.place(x=x, y=y)
            x += w + padding

        self.topbar_inner.configure(height=y + row_height)
    
    def draw_title(self, _event=None) -> None:
        if self.title_canvas is None:
            return
        canvas = self.title_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=WINDOW_BG, outline="")
        draw_round_rect(canvas, 0, 6, width - 2, height - 2, 14, fill=HEADER_BG, outline=PANEL_EDGE, width=1)
        for idx in range(4):
            y = 18 + (idx * 12)
            canvas.create_line(16, y, width - 16, y + (4 if idx % 2 else -3), fill=ACCENT, width=1)
        canvas.create_text(18, 16, anchor="nw", text="WO3 Steel Unit Editor", fill=STATUS_TEXT, font=("Segoe UI", 20, "bold"))
        canvas.create_text(20, 44, anchor="nw", text="Steel table editor with editable field names, grouping/sorting, etc.", fill="#E0E7ED", font=("Segoe UI", 9))

    def set_status(self, text: str, color: str = STATUS_TEXT) -> None:
        self.status_var.set(text)
        if self.status_label is not None:
            self.status_label.configure(fg=color)

    def prepare_header_entries(self) -> None:
        self.field_name_vars.clear()
        self.field_name_entries.clear()
        if self.header_canvas is None:
            return
        for index, name in enumerate(self.field_names):
            var = tk.StringVar(value=name)
            entry = tk.Entry(self.header_canvas, textvariable=var, relief="flat", bd=0, bg=INPUT_BG, fg=INPUT_TEXT, insertbackground=INPUT_TEXT, font=("Segoe UI", 9, "bold"), justify="center")
            entry.bind("<FocusOut>", lambda _e, idx=index: self.commit_field_name(idx))
            entry.bind("<Return>", lambda _e, idx=index: self.commit_field_name(idx))
            entry.bind("<MouseWheel>", self.on_body_mousewheel)
            entry.bind("<Shift-MouseWheel>", self.on_body_shift_mousewheel)
            self.field_name_vars.append(var)
            self.field_name_entries.append(entry)

    def recompute_column_widths(self) -> None:
        widths = []
        for field, name in zip(FIELD_SPECS, self.field_names):
            base = 94 if field.size == 2 else 82
            widths.append(max(base, min(180, len(name) * 7 + 26)))
        self.column_widths = widths
        self.x_offset = clamp(self.x_offset, 0.0, self.max_x_offset())

    def working_block_loaded(self) -> bool:
        return len(self.working_stream.getvalue()) == UNIT_BLOCK_SIZE

    def current_block(self) -> bytes:
        return self.working_stream.getvalue()

    def max_x_offset(self) -> float:
        viewport = max(1.0, (self.body_canvas.winfo_width() if self.body_canvas is not None else 1))
        return max(0.0, float(sum(self.column_widths)) - viewport)

    def max_y_offset(self) -> float:
        if self.body_canvas is None:
            return 0.0
        return max(0.0, (UNIT_SLOT_COUNT * self.row_height) - self.body_canvas.winfo_height())

    def rebuild_view_lookup(self) -> None:
        self.view_lookup = {slot: index for index, slot in enumerate(self.view_order)}

    def rebuild_slot_cache(self) -> None:
        block = self.current_block()
        self.slot_cache = [read_slot_values(block, slot_index) for slot_index in range(UNIT_SLOT_COUNT)]

    def rebuild_changed_rows(self) -> None:
        current = self.current_block()
        self.changed_slots_vs_origin = set()
        if len(current) != UNIT_BLOCK_SIZE or len(self.working_origin_block) != UNIT_BLOCK_SIZE:
            return
        for slot_index in range(UNIT_SLOT_COUNT):
            start = slot_offset_in_block(slot_index)
            end = start + UNIT_SLOT_SIZE
            if current[start:end] != self.working_origin_block[start:end]:
                self.changed_slots_vs_origin.add(slot_index)

    def ensure_slot_visible(self, slot: Optional[int]) -> None:
        if slot is None or self.body_canvas is None or slot not in self.view_lookup:
            return
        row_index = self.view_lookup[slot]
        y1 = row_index * self.row_height
        y2 = y1 + self.row_height
        view_top = self.y_offset
        view_bottom = self.y_offset + self.body_canvas.winfo_height()
        if y1 < view_top:
            self.y_offset = y1
        elif y2 > view_bottom:
            self.y_offset = y2 - self.body_canvas.winfo_height()
        self.y_offset = clamp(self.y_offset, 0.0, self.max_y_offset())

    def update_scrollbars(self) -> None:
        if self.v_scroll is not None:
            max_y = self.max_y_offset()
            if self.body_canvas is None or max_y <= 0:
                self.v_scroll.set_view(0.0, 1.0)
            else:
                total = UNIT_SLOT_COUNT * self.row_height
                first = self.y_offset / max_y
                span = min(0.98, self.body_canvas.winfo_height() / max(1.0, total))
                self.v_scroll.set_view(first, min(1.0, first + span))
        if self.h_scroll is not None:
            max_x = self.max_x_offset()
            if self.body_canvas is None or max_x <= 0:
                self.h_scroll.set_view(0.0, 1.0)
            else:
                viewport = max(1.0, self.body_canvas.winfo_width())
                total = float(sum(self.column_widths))
                first = self.x_offset / max_x
                span = min(0.98, viewport / max(1.0, total))
                self.h_scroll.set_view(first, min(1.0, first + span))

    def current_selection_in_view_order(self) -> List[int]:
        return [slot for slot in self.view_order if slot in self.selected_slot_set]

    def visible_row_span(self) -> Tuple[int, int]:
        if self.body_canvas is None:
            return (0, 0)
        first = max(0, int(self.y_offset // self.row_height))
        visible = int(self.body_canvas.winfo_height() // self.row_height) + 8
        count = min(MAX_DRAWN_ROWS, max(visible, 18))
        return first, min(UNIT_SLOT_COUNT, first + count)

    def ensure_header_pool(self) -> None:
        if self.header_canvas is None:
            return
        canvas = self.header_canvas
        if self.header_background_id is None:
            self.header_background_id = canvas.create_rectangle(0, 0, 0, 0, fill=PANEL_BG, outline="")
            self.header_index_ids = {
                "box": draw_round_rect(canvas, 0, 0, 0, 0, 8, fill=HEADER_BG, outline=PANEL_EDGE, width=1),
                "slot": canvas.create_text(0, 0, text="Slot", fill=STATUS_TEXT, font=("Segoe UI", 9, "bold")),
                "sort_box": draw_round_rect(canvas, 0, 0, 0, 0, 7, fill=HEADER_BG, outline=GRID_LINE, width=1),
                "sort": canvas.create_text(0, 0, text="Sort", fill=ACCENT, font=("Segoe UI", 7, "bold")),
            }
        while len(self.header_column_items) < len(FIELD_SPECS):
            box_id = draw_round_rect(canvas, 0, 0, 0, 0, 8, fill=PANEL_ALT, outline=PANEL_EDGE, width=1)
            window_id = canvas.create_window(0, 0, anchor="nw", width=40, height=22, window=self.field_name_entries[len(self.header_column_items)])
            sort_box = draw_round_rect(canvas, 0, 0, 0, 0, 7, fill=HEADER_BG, outline=GRID_LINE, width=1)
            sort_text = canvas.create_text(0, 0, text="⇅", fill=ACCENT, font=("Segoe UI", 8, "bold"))
            self.header_column_items.append({"box": box_id, "window": window_id, "sort_box": sort_box, "sort_text": sort_text})

    def render_name_body(self):
        if self.body_name_canvas is None:
            return

        canvas = self.body_name_canvas
        canvas.delete("all")

        if not self.working_block_loaded():
            return

        first_row, end_row = self.visible_row_span()

        for display_index in range(first_row, end_row):
            y1 = (display_index * self.row_height) - self.y_offset
            y2 = y1 + self.row_height
            slot_index = self.view_order[display_index]

            values = self.slot_cache[slot_index]
            name_id = values[0]

            name = ""
            if 0 <= name_id < len(self.unit_names):
                name = self.unit_names[name_id]

            selected = slot_index in self.selected_slot_set
            primary = slot_index == self.primary_slot

            if primary:
                fill = PRIMARY_INDEX_FILL
            elif selected:
                fill = "#D7DEE5"
            else:
                fill = "#E6EBF0"

            canvas.create_rectangle(0, y1, self.name_width, y2, fill=fill, outline=GRID_LINE)

            canvas.create_text(
                6, (y1 + y2) / 2,
                anchor="w",
                text=name,
                fill=GRID_TEXT,
                font=("Segoe UI", 9, "bold" if selected or primary else "normal")
            )

    def render_name_header(self):
        if self.header_name_canvas is None:
            return

        canvas = self.header_name_canvas
        canvas.delete("all")

        height = canvas.winfo_height()

        canvas.create_rectangle(0, 0, self.name_width, height, fill=PANEL_BG, outline="")

        canvas.create_polygon(
            rounded_rect_points(0, 0, self.name_width - 1, height - 1, 8),
            fill=HEADER_BG,
            outline=PANEL_EDGE
        )

        canvas.create_text(
            self.name_width / 2,
            20,
            text="Unit Name",
            fill=STATUS_TEXT,
            font=("Segoe UI", 9, "bold")
        )
    
    def render_header(self) -> None:
        if self.header_canvas is None or self.header_index_canvas is None:
            return
        index_canvas = self.header_index_canvas
        index_canvas.delete("all")

        height = max(1, index_canvas.winfo_height())
        index_canvas.create_rectangle(
            0, 0, self.index_width, height,
            fill=PANEL_BG, outline=""
        )
        index_canvas.create_polygon(
            rounded_rect_points(0, 0, self.index_width - 1, height - 1, 8),
            fill=HEADER_BG,
            outline=PANEL_EDGE
        )
        index_canvas.create_text(
            self.index_width / 2, 20,
            text="Slot",
            fill=STATUS_TEXT,
            font=("Segoe UI", 9, "bold")
        )
        sort_x1 = 8
        sort_y1 = 39
        sort_x2 = self.index_width - 10
        sort_y2 = height - 8

        slot_active = self.sort_field is None
        slot_symbol = "^" if slot_active else "<>"

        index_canvas.create_polygon(
            rounded_rect_points(sort_x1, sort_y1, sort_x2, sort_y2, 7),
            fill=HEADER_BG,
            outline=ACCENT if slot_active else GRID_LINE
        )

        index_canvas.create_text(
            (sort_x1 + sort_x2) / 2,
            (sort_y1 + sort_y2) / 2,
            text=slot_symbol,
            fill=STATUS_TEXT if slot_active else ACCENT,
            font=("Segoe UI", 8, "bold")
        )

        self.slot_sort_box = (sort_x1, sort_y1, sort_x2, sort_y2)
        canvas = self.header_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())

        self.ensure_header_pool()
        self.header_sort_boxes.clear()

        if self.header_background_id is not None:
            canvas.coords(self.header_background_id, 0, 0, width, height)
            canvas.itemconfigure(self.header_background_id, fill=PANEL_BG, outline="")
            canvas.tag_lower(self.header_background_id)

        x = -self.x_offset

        for index, width_px in enumerate(self.column_widths):
            column = self.header_column_items[index]

            if x + width_px < -6:
                for item_id in column.values():
                    canvas.itemconfigure(item_id, state="hidden")
                x += width_px
                continue

            if x > width + 6:
                for item_id in column.values():
                    canvas.itemconfigure(item_id, state="hidden")
                x += width_px
                continue

            canvas.coords(column["box"], *rounded_rect_points(x, 1, x + width_px - 2, height - 1, 8))
            canvas.itemconfigure(column["box"], fill=PANEL_ALT, outline=PANEL_EDGE, state="normal")

            canvas.coords(column["window"], x + 6, 16)
            canvas.itemconfigure(column["window"], width=max(50, width_px - 12), height=22, state="normal")

            sort_x1 = x + 8
            sort_y1 = 39
            sort_x2 = x + width_px - 10
            sort_y2 = height - 8

            active = self.sort_field == index
            symbol = "▼" if active and self.sort_reverse else ("▲" if active else "⇅")

            canvas.coords(column["sort_box"], *rounded_rect_points(sort_x1, sort_y1, sort_x2, sort_y2, 7))
            canvas.itemconfigure(column["sort_box"], fill=HEADER_BG, outline=ACCENT if active else GRID_LINE, state="normal")

            canvas.coords(column["sort_text"], (sort_x1 + sort_x2) / 2, (sort_y1 + sort_y2) / 2)
            canvas.itemconfigure(column["sort_text"], text=symbol, fill=STATUS_TEXT if active else ACCENT, state="normal")

            self.header_sort_boxes.append((index, (sort_x1, sort_y1, sort_x2, sort_y2)))

            x += width_px

        self.update_scrollbars()

    def ensure_body_background(self, width: float, height: float) -> None:
        if self.body_canvas is None:
            return
        canvas = self.body_canvas
        if self.body_background_id is None:
            self.body_background_id = canvas.create_rectangle(0, 0, width, height, fill=GRID_BG, outline="")
        else:
            canvas.coords(self.body_background_id, 0, 0, width, height)
            canvas.itemconfigure(self.body_background_id, fill=GRID_BG, outline="")
        canvas.tag_lower(self.body_background_id)

    def clear_body_pool(self) -> None:
        if self.body_canvas is None:
            return
        if self.body_empty_text_id is not None:
            self.body_canvas.delete(self.body_empty_text_id)
            self.body_empty_text_id = None
        for row in self.body_row_items:
            for item_id in row["all_ids"]:
                self.body_canvas.delete(item_id)
        self.body_row_items.clear()

    def ensure_body_row_pool(self, count: int) -> None:
        if self.body_canvas is None:
            return
        canvas = self.body_canvas
        while len(self.body_row_items) < count:
            row_rect = canvas.create_rectangle(0, 0, 0, 0, fill=GRID_BG, outline=GRID_LINE)
            index_rect = canvas.create_rectangle(0, 0, 0, 0, fill="#E6EBF0", outline=GRID_LINE)
            index_text = canvas.create_text(0, 0, anchor="w", text="", fill=GRID_TEXT, font=("Consolas", 9))
            changed_marker = canvas.create_rectangle(0, 0, 0, 0, fill="#8092A4", outline="", state="hidden")
            selection_rect = canvas.create_rectangle(0, 0, 0, 0, outline=SELECT_EDGE, width=2, state="hidden")
            cell_rects = []
            cell_texts = []
            all_ids = [row_rect, index_rect, index_text, changed_marker, selection_rect]
            for _field in FIELD_SPECS:
                rect_id = canvas.create_rectangle(0, 0, 0, 0, outline=GRID_LINE, fill=GRID_BG)
                text_id = canvas.create_text(0, 0, anchor="w", text="", fill=GRID_TEXT, font=("Segoe UI", 9))
                cell_rects.append(rect_id)
                cell_texts.append(text_id)
                all_ids.extend([rect_id, text_id])
            self.body_row_items.append(
                {
                    "row_rect": row_rect,
                    "index_rect": index_rect,
                    "index_text": index_text,
                    "changed_marker": changed_marker,
                    "selection_rect": selection_rect,
                    "cell_rects": cell_rects,
                    "cell_texts": cell_texts,
                    "all_ids": all_ids,
                }
            )
        for row in self.body_row_items:
            for item_id in row["all_ids"]:
                canvas.itemconfigure(item_id, state="hidden")
        if self.body_background_id is not None:
            canvas.tag_lower(self.body_background_id)

    def render_body(self) -> None:
        if self.body_canvas is None:
            return
        canvas = self.body_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        self.ensure_body_background(width, height)
        if not self.working_block_loaded():
            self.clear_body_pool()
            if self.body_empty_text_id is None:
                self.body_empty_text_id = canvas.create_text(width / 2, height / 2, text="Open LINKFILE_000.bin or a saved unit mod to begin.", fill=GRID_MUTED, font=("Segoe UI", 12, "bold"))
            else:
                canvas.coords(self.body_empty_text_id, width / 2, height / 2)
                canvas.itemconfigure(self.body_empty_text_id, text="Open LINKFILE_000.bin or a saved unit mod to begin.", state="normal")
            self.update_scrollbars()
            return

        if self.body_empty_text_id is not None:
            canvas.delete(self.body_empty_text_id)
            self.body_empty_text_id = None
        first_row, end_row = self.visible_row_span()
        visible_count = max(0, end_row - first_row)
        self.ensure_body_row_pool(visible_count)
        for pool_index, display_index in enumerate(range(first_row, end_row)):
            row = self.body_row_items[pool_index]
            y1 = (display_index * self.row_height) - self.y_offset
            y2 = y1 + self.row_height
            slot_index = self.view_order[display_index]
            selected = slot_index in self.selected_slot_set
            primary = slot_index == self.primary_slot
            changed = slot_index in self.changed_slots_vs_origin
            if primary:
                fill = PRIMARY_SELECT_FILL
            elif selected:
                fill = SELECT_FILL
            else:
                fill = GRID_ALT if display_index % 2 else GRID_BG
            data_x1 = self.index_width
            data_x2 = width

            canvas.coords(row["row_rect"], data_x1, y1, data_x2, y2)
            canvas.itemconfigure(row["row_rect"], fill=fill, outline=GRID_LINE, state="normal")
            index_fill = PRIMARY_INDEX_FILL if primary else ("#D7DEE5" if selected else "#E6EBF0")
            text_style = "bold" if selected or primary else "normal"
            if changed:
                canvas.coords(row["changed_marker"], self.index_width - 6, y1 + 6, self.index_width - 2, y2 - 6)
                canvas.itemconfigure(row["changed_marker"], state="normal")
            else:
                canvas.itemconfigure(row["changed_marker"], state="hidden")

            x = -self.x_offset
            values = self.slot_cache[slot_index]
            cell_rects = row["cell_rects"]
            cell_texts = row["cell_texts"]
            for field_index, width_px in enumerate(self.column_widths):
                rect_id = cell_rects[field_index]
                text_id = cell_texts[field_index]
                canvas.coords(rect_id, x, y1, x + width_px, y2)
                canvas.itemconfigure(rect_id, outline=GRID_LINE, fill=fill, state="normal")
                padding = 3 if field_index == 0 else 6
                canvas.coords(text_id, x + padding, (y1 + y2) / 2)
                value = values[field_index]
                field_name = FIELD_SPECS[field_index].name

                display_text = str(value)

                if field_name == "Voice ID":
                    name = self.voice_names.get(value, "")
                    if name:
                        display_text = f"{value} ({self.clip_text_to_width(name, width_px)})"

                elif field_name == "Model ID":
                    name = self.model_names.get(value, "")
                    if name:
                        display_text = f"{value} ({self.clip_text_to_width(name, width_px)})"

                elif field_name == "Moveset":
                    name = self.moveset_names.get(value, "")
                    if name:
                        display_text = f"{value} ({self.clip_text_to_width(name, width_px)})"

                elif field_name == "Name":
                    if 0 <= value < len(self.unit_names):
                        name = self.unit_names[value]

                        base = f"{value} ("
                        suffix = ")"

                        avg_char_width = 7
                        reserved = len(base + suffix) * avg_char_width
                        available = max(10, width_px - reserved)

                        clipped = self.clip_text_to_width(name, available)

                        display_text = f"{base}{clipped}{suffix}"

                canvas.itemconfigure(
                    text_id,
                    text=display_text,
                    fill=GRID_TEXT,
                    font=("Segoe UI", 9),
                    state="normal"
                )
                x += width_px
            if selected:
                canvas.coords(
                    row["selection_rect"],
                    self.index_width + 1,
                    y1 + 1,
                    width - 2,
                    y2 - 1
                )
                canvas.itemconfigure(row["selection_rect"], outline="#6C87A8" if primary else SELECT_EDGE, width=2, state="normal")
            else:
                canvas.itemconfigure(row["selection_rect"], state="hidden")

        for pool_index in range(visible_count, len(self.body_row_items)):
            row = self.body_row_items[pool_index]
            for item_id in row["all_ids"]:
                canvas.itemconfigure(item_id, state="hidden")
        self.update_scrollbars()

    def render_table(self):
        self.render_header()
        self.render_name_header()
        
        self.render_body()
        self.render_name_body()
        
        self.render_index_body()

    def ensure_multi_editor(self) -> None:
        if self.multi_editor_window is not None and self.multi_editor_window.winfo_exists():
            return
        window = tk.Toplevel(self)
        window.withdraw()
        window.title(f"{APP_TITLE} - Multi Edit")
        window.configure(bg=WINDOW_BG)
        window.transient(self)
        window.minsize(980, 220)
        window.protocol("WM_DELETE_WINDOW", self.hide_multi_editor)

        header = tk.Frame(window, bg=HEADER_BG, highlightthickness=1, highlightbackground=PANEL_EDGE)
        header.pack(fill="x", padx=12, pady=(12, 8))
        tk.Label(header, textvariable=self.multi_editor_title_var, bg=HEADER_BG, fg=STATUS_TEXT, anchor="w", font=("Segoe UI", 13, "bold")).pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(header, textvariable=self.multi_editor_info_var, bg=HEADER_BG, fg="#DCE6EE", anchor="w", font=("Segoe UI", 9)).pack(fill="x", padx=12, pady=(0, 10))

        table_shell = tk.Frame(window, bg=WINDOW_BG)
        table_shell.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.multi_editor_canvas = tk.Canvas(table_shell, bg=GRID_BG, highlightthickness=1, highlightbackground=PANEL_EDGE, bd=0, relief="flat", height=88)
        self.multi_editor_canvas.pack(fill="both", expand=True)
        self.multi_editor_canvas.bind("<Configure>", lambda _e: self.layout_multi_editor_fields())
        self.multi_editor_canvas.bind("<MouseWheel>", self.on_multi_editor_mousewheel)
        self.multi_editor_canvas.bind("<Shift-MouseWheel>", self.on_multi_editor_mousewheel)
        self.multi_editor_inner = tk.Frame(self.multi_editor_canvas, bg=GRID_BG, width=100, height=88)
        self.multi_editor_window_id = self.multi_editor_canvas.create_window(0, 0, anchor="nw", window=self.multi_editor_inner)

        self.multi_editor_vars.clear()
        self.multi_editor_labels.clear()
        self.multi_editor_entries.clear()
        self.multi_editor_mixed = [False for _ in FIELD_SPECS]
        for index, _field in enumerate(FIELD_SPECS):
            var = tk.StringVar(value="")
            label = tk.Label(self.multi_editor_inner, text=self.field_names[index], bg=HEADER_BG, fg=STATUS_TEXT, anchor="center", font=("Segoe UI", 9, "bold"), highlightthickness=1, highlightbackground=PANEL_EDGE, bd=0, relief="flat")
            entry = tk.Entry(self.multi_editor_inner, textvariable=var, relief="flat", bd=0, bg=INPUT_BG, fg=INPUT_TEXT, insertbackground=INPUT_TEXT, justify="center", font=("Consolas", 10))
            entry.bind("<Return>", lambda _e: self.apply_multi_editor_changes())
            self.multi_editor_vars.append(var)
            self.multi_editor_labels.append(label)
            self.multi_editor_entries.append(entry)

        self.multi_editor_scroll = SteelKnobScrollbar(window, orient="horizontal")
        self.multi_editor_scroll.pack(fill="x", padx=12, pady=(0, 8))
        self.multi_editor_scroll.set_command(self.on_multi_editor_scroll)

        actions = tk.Frame(window, bg=WINDOW_BG)
        actions.pack(fill="x", padx=12, pady=(0, 12))
        SteelButton(actions, "Apply To Selection", self.apply_multi_editor_changes, width=150, height=30).pack(side="left")
        SteelButton(actions, "Reload", self.reload_multi_editor_snapshot, width=86, height=30).pack(side="left", padx=(8, 0))
        SteelButton(actions, "Close", self.hide_multi_editor, width=86, height=30).pack(side="right")

        self.multi_editor_window = window

    def layout_multi_editor_fields(self) -> None:
        if self.multi_editor_canvas is None or self.multi_editor_inner is None or self.multi_editor_window_id is None:
            return
        x = 0
        for index, width_px in enumerate(self.column_widths):
            cell_width = max(82, width_px)
            self.multi_editor_labels[index].configure(text=self.field_names[index])
            self.multi_editor_labels[index].place(x=x + 2, y=6, width=cell_width - 4, height=24)
            self.multi_editor_entries[index].place(x=x + 2, y=38, width=cell_width - 4, height=28)
            x += cell_width
        content_width = max(1, x)
        content_height = 72
        self.multi_editor_inner.configure(width=content_width, height=content_height)
        self.multi_editor_canvas.itemconfigure(self.multi_editor_window_id, width=content_width, height=content_height)
        self.multi_editor_canvas.configure(scrollregion=(0, 0, content_width, content_height))
        self.update_multi_editor_scrollbar()

    def position_multi_editor(self) -> None:
        if self.multi_editor_window is None or not self.multi_editor_window.winfo_exists():
            return
        self.update_idletasks()
        self.multi_editor_window.update_idletasks()
        width = max(980, min(1500, self.winfo_width() - 80))
        height = 270
        x = self.winfo_rootx() + max(30, (self.winfo_width() - width) // 2)
        y = self.winfo_rooty() + 90
        self.multi_editor_window.geometry(f"{width}x{height}+{x}+{y}")

    def update_multi_editor_scrollbar(self) -> None:
        if self.multi_editor_canvas is None or self.multi_editor_scroll is None:
            return
        self.multi_editor_canvas.update_idletasks()
        viewport = max(1.0, self.multi_editor_canvas.winfo_width())
        bbox = self.multi_editor_canvas.bbox(self.multi_editor_window_id) if self.multi_editor_window_id is not None else None
        content_width = max(viewport, float((bbox[2] - bbox[0]) if bbox else 1.0))
        span = min(1.0, viewport / max(1.0, content_width))
        max_first = max(0.0, 1.0 - span)
        first, _last = self.multi_editor_canvas.xview()
        if max_first <= 0.0:
            self.multi_editor_scroll.set_view(0.0, 1.0)
            return
        first_norm = first / max_first
        self.multi_editor_scroll.set_view(first_norm, min(1.0, first_norm + span))

    def on_multi_editor_scroll(self, fraction: float) -> None:
        if self.multi_editor_canvas is None:
            return
        viewport = max(1.0, self.multi_editor_canvas.winfo_width())
        bbox = self.multi_editor_canvas.bbox(self.multi_editor_window_id) if self.multi_editor_window_id is not None else None
        content_width = max(viewport, float((bbox[2] - bbox[0]) if bbox else 1.0))
        span = min(1.0, viewport / max(1.0, content_width))
        max_first = max(0.0, 1.0 - span)
        self.multi_editor_canvas.xview_moveto(0.0 if max_first <= 0.0 else clamp(fraction, 0.0, 1.0) * max_first)
        self.update_multi_editor_scrollbar()

    def on_multi_editor_mousewheel(self, event):
        if self.multi_editor_canvas is None:
            return "break"
        first, last = self.multi_editor_canvas.xview()
        span = max(0.0, last - first)
        limit = max(0.0, 1.0 - span)
        step = 0.08
        delta = -step if event.delta > 0 else step
        self.multi_editor_canvas.xview_moveto(clamp(first + delta, 0.0, limit))
        self.update_multi_editor_scrollbar()
        return "break"

    def hide_multi_editor(self) -> None:
        if self.multi_editor_window is not None and self.multi_editor_window.winfo_exists():
            self.multi_editor_window.withdraw()

    def capture_multi_editor_snapshot(self) -> None:
        snapshot_slots = sorted(self.selected_slots)
        self.multi_editor_snapshot_slots = snapshot_slots
        self.multi_editor_snapshot_values = {slot: self.slot_cache[slot][:] for slot in snapshot_slots}

    def refresh_multi_editor_from_selection(self, *, capture_snapshot: bool = False) -> None:
        if len(self.selected_slots) <= 1 or not self.working_block_loaded():
            self.multi_editor_snapshot_slots = []
            self.multi_editor_snapshot_values = {}
            self.hide_multi_editor()
            return
        self.ensure_multi_editor()
        current_slots = sorted(self.selected_slots)
        if capture_snapshot or self.multi_editor_snapshot_slots != current_slots:
            self.capture_multi_editor_snapshot()
        self.multi_editor_title_var.set(f"Multi-Slot Editor | {len(self.selected_slots)} slots selected")
        self.multi_editor_info_var.set(f"Slots: {summarize_ranges(self.selected_slots, limit=8)}")
        self.multi_editor_mixed = []
        for index, field in enumerate(FIELD_SPECS):
            values = [self.slot_cache[slot][index] for slot in self.selected_slots]
            mixed = any(value != values[0] for value in values[1:])
            self.multi_editor_mixed.append(mixed)
            self.multi_editor_vars[index].set(MIXED_VALUE_TEXT if mixed else format_field_value(field, values[0]))
            self.multi_editor_labels[index].configure(text=self.field_names[index])
            self.multi_editor_entries[index].configure(bg="#F1F4F7" if mixed else INPUT_BG, fg=INPUT_TEXT)
        self.layout_multi_editor_fields()
        self.position_multi_editor()
        if self.multi_editor_window is not None:
            self.multi_editor_window.deiconify()
            self.multi_editor_window.lift()

    def sync_multi_editor_with_selection(self) -> None:
        if len(self.selected_slots) > 1 and self.working_block_loaded():
            self.refresh_multi_editor_from_selection(capture_snapshot=self.multi_editor_snapshot_slots != sorted(self.selected_slots))
        else:
            self.multi_editor_snapshot_slots = []
            self.multi_editor_snapshot_values = {}
            self.hide_multi_editor()

    def apply_multi_editor_changes(self) -> None:
        if len(self.selected_slots) <= 1:
            self.hide_multi_editor()
            return
        updates: List[Tuple[int, int]] = []
        for index, field in enumerate(FIELD_SPECS):
            raw = self.multi_editor_vars[index].get().strip()
            if self.multi_editor_mixed[index] and raw == MIXED_VALUE_TEXT:
                continue
            try:
                value = parse_field_text(raw, field.size)
            except ValueError as exc:
                messagebox.showerror("Invalid Field Value", f"{self.field_names[index]}: {exc}")
                self.set_status(f"{self.field_names[index]} is invalid.", BAD)
                return
            updates.append((index, value))
        if not updates:
            self.set_status("No multi-edit changes to apply.", WARN)
            return
        self.apply_field_updates_to_slots(updates, self.selected_slots, status_text=f"Updated {len(updates)} field{'s' if len(updates) != 1 else ''} across {len(self.selected_slots)} selected slots.")
        self.refresh_multi_editor_from_selection(capture_snapshot=False)

    def reload_multi_editor_snapshot(self) -> None:
        if len(self.multi_editor_snapshot_slots) <= 1 or not self.multi_editor_snapshot_values:
            self.set_status("No multi-slot snapshot is ready to reload.", WARN)
            return
        self.restore_slots_from_snapshot(
            self.multi_editor_snapshot_values,
            self.multi_editor_snapshot_slots,
            status_text=f"Reloaded original values for {len(self.multi_editor_snapshot_slots)} selected slots.",
        )
    def render_index_body(self) -> None:
        if self.body_index_canvas is None:
            return

        canvas = self.body_index_canvas
        canvas.delete("all")

        height = max(1, canvas.winfo_height())

        if not self.working_block_loaded():
            return

        first_row, end_row = self.visible_row_span()

        for display_index in range(first_row, end_row):
            y1 = (display_index * self.row_height) - self.y_offset
            y2 = y1 + self.row_height
            slot_index = self.view_order[display_index]

            selected = slot_index in self.selected_slot_set
            primary = slot_index == self.primary_slot

            if primary:
                fill = PRIMARY_INDEX_FILL
            elif selected:
                fill = "#D7DEE5"
            else:
                fill = "#E6EBF0"

            canvas.create_rectangle(0, y1, self.index_width, y2, fill=fill, outline=GRID_LINE)

            canvas.create_text(
                8, (y1 + y2) / 2,
                anchor="w",
                text=str(slot_index),
                fill=GRID_TEXT,
                font=("Consolas", 9, "bold" if selected or primary else "normal")
            )
        

    def on_vertical_scroll(self, fraction: float) -> None:
        self.y_offset = clamp(fraction * self.max_y_offset(), 0.0, self.max_y_offset())
        self.render_body()
        self.render_index_body()
        self.render_name_body()
        

    def on_horizontal_scroll(self, fraction: float) -> None:
        self.x_offset = clamp(fraction * self.max_x_offset(), 0.0, self.max_x_offset())
        self.render_table()

    def on_body_mousewheel(self, event):
        self.y_offset = clamp(
            self.y_offset + ((-3 if event.delta > 0 else 3) * self.row_height), 
            0.0, 
            self.max_y_offset()
        )
        self.render_body()
        self.render_index_body()
        self.render_name_body()
        
        self.update_scrollbars()
        return "break"

    def on_body_shift_mousewheel(self, event):
        self.x_offset = clamp(self.x_offset + (-120 if event.delta > 0 else 120), 0.0, self.max_x_offset())
        self.render_table()
        return "break"

    def body_hit_test(self, x: float, y: float) -> Tuple[Optional[int], Optional[int]]:
        if self.body_canvas is None or not self.working_block_loaded():
            return None, None

        row = int((y + self.y_offset) // self.row_height)
        if row < 0 or row >= len(self.view_order):
            return None, None

        slot_index = self.view_order[row]

        local_x = x + self.x_offset

        running = 0.0
        for field_index, width_px in enumerate(self.column_widths):
            if running <= local_x < running + width_px:
                return slot_index, field_index
            running += width_px

        return slot_index, None

    def set_selected_slots(self, slots: Sequence[int], *, primary_slot: Optional[int] = None) -> None:
        valid_set = {slot for slot in slots if 0 <= slot < UNIT_SLOT_COUNT}
        self.selected_slot_set = valid_set
        self.selected_slots = [slot for slot in self.view_order if slot in valid_set]
        if self.selected_slots:
            active = primary_slot if primary_slot in valid_set else self.selected_slots[-1]
            self.primary_slot = active
            self.selection_anchor = active
            self.ensure_slot_visible(active)
        else:
            self.primary_slot = None
            self.selection_anchor = None
        self.close_inline_editor(commit=False)
        self.render_body()
        self.sync_multi_editor_with_selection()

    def toggle_slot(self, slot: int) -> None:
        if slot in self.selected_slot_set:
            new_slots = [value for value in self.selected_slots if value != slot]
            primary = self.primary_slot if self.primary_slot in new_slots else (new_slots[-1] if new_slots else None)
        else:
            new_slots = self.selected_slots + [slot]
            primary = slot
        self.set_selected_slots(new_slots, primary_slot=primary)

    def on_body_click(self, event) -> None:
        slot_index, _field_index = self.body_hit_test(event.x, event.y)
        self.close_inline_editor(commit=True)
        if slot_index is None:
            return
        if event.state & SHIFT_MASK and self.selection_anchor is not None and self.selection_anchor in self.view_lookup:
            start = min(self.view_lookup[self.selection_anchor], self.view_lookup[slot_index])
            end = max(self.view_lookup[self.selection_anchor], self.view_lookup[slot_index])
            self.set_selected_slots(self.view_order[start : end + 1], primary_slot=slot_index)
        elif event.state & CTRL_MASK:
            self.toggle_slot(slot_index)
        else:
            self.set_selected_slots([slot_index], primary_slot=slot_index)

    def cell_bounds(self, slot_index: int, field_index: int) -> Optional[Tuple[float, float, float, float]]:
        if self.body_canvas is None or slot_index not in self.view_lookup:
            return None
        row = self.view_lookup[slot_index]
        y1 = (row * self.row_height) - self.y_offset
        y2 = y1 + self.row_height
        if y2 < 0 or y1 > self.body_canvas.winfo_height():
            return None
        x1 = float(self.index_width) - self.x_offset + sum(self.column_widths[:field_index])
        x2 = x1 + self.column_widths[field_index]
        return (x1, y1, x2, y2)

    def inline_edit_targets(self, slot_index: int) -> List[int]:
        if slot_index in self.selected_slot_set and self.selected_slots:
            return self.selected_slots[:]
        return [slot_index]

    def inline_display_value(self, slot_index: int, field_index: int) -> Tuple[str, bool]:
        targets = self.inline_edit_targets(slot_index)
        values = [self.slot_cache[target][field_index] for target in targets]
        if values and all(value == values[0] for value in values[1:]):
            return format_field_value(FIELD_SPECS[field_index], values[0]), False
        return MIXED_VALUE_TEXT, True

    def on_body_double_click(self, event) -> None:
        slot_index, field_index = self.body_hit_test(event.x, event.y)
        if slot_index is None or field_index is None:
            return
        if len(self.selected_slots) > 1 and slot_index in self.selected_slot_set:
            self.refresh_multi_editor_from_selection()
            return
        if slot_index not in self.selected_slot_set:
            self.set_selected_slots([slot_index], primary_slot=slot_index)
        elif self.primary_slot != slot_index:
            self.primary_slot = slot_index
            self.selection_anchor = slot_index
            self.ensure_slot_visible(slot_index)
            self.render_body()
        self.open_inline_editor(slot_index, field_index)

    def open_inline_editor(self, slot_index: int, field_index: int) -> None:
        if self.body_canvas is None:
            return
        bounds = self.cell_bounds(slot_index, field_index)
        if bounds is None:
            return
        self.close_inline_editor(commit=False)
        x1, y1, x2, y2 = bounds
        display_value, mixed = self.inline_display_value(slot_index, field_index)
        self.inline_edit_context = (slot_index, field_index, mixed)
        field = FIELD_SPECS[field_index]
        field_name = field.name

        options = None

        if field_name == "Voice ID":
            options = self.voice_names
        elif field_name == "Model ID":
            options = self.model_names
        elif field_name == "Moveset":
            options = self.moveset_names
        elif field_name == "Name":
            options = {i: name for i, name in enumerate(self.unit_names)}

        if self.inline_editor is not None:
            try:
                self.inline_editor.destroy()
            except:
                pass
            self.inline_editor = None

        if options:
            from tkinter import ttk

            values = [f"{k} ({v})" for k, v in sorted(options.items())]

            self.inline_editor = ttk.Combobox(
                self.body_canvas,
                values=values,
                state="normal",
                font=("Consolas", 10)
            )

            try:
                val = int(display_value.split(" ")[0])
                if val in options:
                    self.inline_editor.set(f"{val} ({options[val]})")
                else:
                    self.inline_editor.set(display_value)
            except:
                self.inline_editor.set(display_value)

            def commit_dropdown(event=None):
                text = self.inline_editor.get().strip()
                try:
                    new_val = int(text.split(" ", 1)[0])
                except:
                    return

                targets = self.selected_slots[:] if self.selected_slots else [slot_index]
                self.apply_value_to_slots(field_index, new_val, targets)

                self.close_inline_editor(commit=False)
                self.render_table()

            self.inline_editor.bind("<<ComboboxSelected>>", commit_dropdown)
            self.inline_editor.bind("<Return>", commit_dropdown)
            self.inline_editor.bind("<FocusOut>", lambda e: commit_dropdown())

        else:
            self.inline_editor = tk.Entry(
                self.body_canvas,
                textvariable=self.inline_edit_var,
                relief="flat",
                bd=0,
                bg=INPUT_BG,
                fg=INPUT_TEXT,
                insertbackground=INPUT_TEXT,
                font=("Consolas", 10)
            )

            self.inline_edit_var.set(display_value)

            self.inline_editor.bind("<Return>", lambda _e: self.close_inline_editor(commit=True))
            self.inline_editor.bind("<Escape>", lambda _e: self.close_inline_editor(commit=False))
            self.inline_editor.bind("<FocusOut>", lambda _e: self.close_inline_editor(commit=True))
        self.inline_edit_var.set(display_value)
        self.body_canvas.create_window(x1 + 4, y1 + 3, anchor="nw", width=max(30, x2 - x1 - 8), height=max(20, y2 - y1 - 6), window=self.inline_editor, tags=("inline_editor",))
        self.inline_editor.focus_set()
        try:
            self.inline_editor.selection_range(0, "end")
        except tk.TclError:
            pass

    def close_inline_editor(self, *, commit: bool) -> None:
        if self.body_canvas is not None:
            self.body_canvas.delete("inline_editor")
        if not commit or self.inline_edit_context is None:
            self.inline_edit_context = None
            return
        try:
            slot_index, field_index, mixed = self.inline_edit_context
            raw_text = self.inline_edit_var.get().strip()
            if mixed and raw_text == MIXED_VALUE_TEXT:
                self.inline_edit_context = None
                self.render_body()
                return
            value = parse_field_text(raw_text, FIELD_SPECS[field_index].size)
        except ValueError as exc:
            messagebox.showerror("Invalid Field Value", str(exc))
            self.set_status(str(exc), BAD)
            self.inline_edit_context = None
            self.render_body()
            return
        targets = self.selected_slots[:] if self.selected_slots else [slot_index]
        self.apply_value_to_slots(field_index, value, targets)
        self.inline_edit_context = None

    def finalize_block_update(self, block: bytearray, *, status_text: str) -> None:
        self.working_stream = io.BytesIO(bytes(block))
        self.rebuild_changed_rows()

        if self.sort_field is not None:
            self.view_order = sorted(
                self.original_order,
                key=lambda slot: self.slot_cache[slot][self.sort_field],
                reverse=self.sort_reverse,
            )
        else:
            self.view_order = self.original_order[:]

        self.rebuild_view_lookup()

        self.selected_slots = self.current_selection_in_view_order()

        self.ensure_slot_visible(self.primary_slot)
        self.render_body()
        self.sync_multi_editor_with_selection()
        self.set_status(status_text, GOOD)

    def apply_field_updates_to_slots(self, updates: Sequence[Tuple[int, int]], slots: Sequence[int], *, status_text: Optional[str] = None) -> None:
        if not self.working_block_loaded():
            return
        block = bytearray(self.current_block())
        for field_index, value in updates:
            field = FIELD_SPECS[field_index]
            raw = value.to_bytes(field.size, "little", signed=False)
            for slot_index in slots:
                start = slot_offset_in_block(slot_index) + field.offset
                block[start : start + field.size] = raw
                self.slot_cache[slot_index][field_index] = value
        if status_text is None:
            if len(updates) == 1:
                field = FIELD_SPECS[updates[0][0]]
                status_text = f"Updated {field.name} for {len(slots)} slot{'s' if len(slots) != 1 else ''}."
            else:
                status_text = f"Updated {len(updates)} fields for {len(slots)} slot{'s' if len(slots) != 1 else ''}."
        self.finalize_block_update(block, status_text=status_text)

    def restore_slots_from_snapshot(self, snapshot_values: Dict[int, List[int]], slots: Sequence[int], *, status_text: str) -> None:
        if not self.working_block_loaded():
            return
        block = bytearray(self.current_block())
        restored = 0
        for slot_index in slots:
            values = snapshot_values.get(slot_index)
            if not values or len(values) != len(FIELD_SPECS):
                continue
            restored += 1
            for field_index, field in enumerate(FIELD_SPECS):
                value = values[field_index]
                start = slot_offset_in_block(slot_index) + field.offset
                block[start : start + field.size] = value.to_bytes(field.size, "little", signed=False)
                self.slot_cache[slot_index][field_index] = value
        if restored == 0:
            self.set_status("No saved multi-slot snapshot was available to reload.", WARN)
            return
        self.finalize_block_update(block, status_text=status_text)

    def apply_value_to_slots(self, field_index: int, value: int, slots: Sequence[int]) -> None:
        self.apply_field_updates_to_slots([(field_index, value)], slots)

    def commit_field_name(self, field_index: int) -> None:
        raw = self.field_name_vars[field_index].get().strip()[:FIELD_NAME_LIMIT]
        self.field_names[field_index] = raw or DEFAULT_FIELD_NAMES[field_index]
        self.field_name_vars[field_index].set(self.field_names[field_index])
        old_widths = self.column_widths[:]

        self.recompute_column_widths()

        if old_widths and len(old_widths) == len(FIELD_SPECS):
            self.column_widths = old_widths
        try:
            save_field_names_json(FIELD_NAMES_JSON, self.field_names, self.column_widths)
            self.set_status("Saved field names.", GOOD)
        except OSError as exc:
            self.set_status(f"Could not save field names: {exc}", BAD)
        self.render_table()
        self.sync_multi_editor_with_selection()

    def export_field_names(self) -> None:
        path = filedialog.asksaveasfilename(title="Export Field Names JSON", initialdir=APP_DATA_DIR, initialfile="wo3_field_names.json", defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            save_field_names_json(path, self.field_names, self.column_widths)
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))
            self.set_status("Could not export field names.", BAD)
            return
        self.set_status(f"Exported field names to {os.path.basename(path)}.", GOOD)

    def import_field_names(self) -> None:
        path = filedialog.askopenfilename(title="Import Field Names JSON", initialdir=APP_DATA_DIR, filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            names, widths = load_field_names_json(path)
        except Exception as exc:
            messagebox.showerror("Import Failed", str(exc))
            self.set_status("Could not import field names.", BAD)
            return
        self.field_names = names
        for index, value in enumerate(names):
            self.field_name_vars[index].set(value)
        self.recompute_column_widths()
        if widths and len(widths) == len(FIELD_SPECS):
            self.column_widths = widths
        save_field_names_json(FIELD_NAMES_JSON, self.field_names, self.column_widths)
        self.render_table()
        self.sync_multi_editor_with_selection()
        self.set_status(f"Imported field names from {os.path.basename(path)}.", GOOD)

    def on_header_click(self, event) -> None:
        if self.slot_sort_box is not None:
            x1, y1, x2, y2 = self.slot_sort_box
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.clear_sort()
                return
        for field_index, (x1, y1, x2, y2) in self.header_sort_boxes:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.toggle_sort(field_index)
                return

    def toggle_sort(self, field_index: int) -> None:
        if not self.slot_cache:
            self.set_status("Load a BIN or mod before sorting.", WARN)
            return
        if self.sort_field == field_index:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_field = field_index
            self.sort_reverse = False
        self.view_order = sorted(self.original_order, key=lambda slot: self.slot_cache[slot][field_index], reverse=self.sort_reverse)
        self.rebuild_view_lookup()
        self.y_offset = 0.0
        self.ensure_slot_visible(self.primary_slot)
        self.render_table()
        name = self.field_names[field_index]
        direction = "descending" if self.sort_reverse else "ascending"
        self.set_status(f"Grouped slots by {name} ({direction}).", GOOD)

    def clear_sort(self) -> None:
        self.sort_field = None
        self.sort_reverse = False

        self.view_order = self.original_order[:]
        self.rebuild_view_lookup()

        self.ensure_slot_visible(self.primary_slot)
        self.render_table()
        self.set_status("Restored original slot order.", GOOD)

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

    def load_block_into_memory(self, block: bytes, *, source_path: Optional[str], compare_block: Optional[bytes]) -> None:
        self.working_stream = io.BytesIO(block)
        self.working_origin_block = bytes(block)
        self.current_mod_path = source_path
        self.compare_block = compare_block if compare_block is not None else block
        self.rebuild_slot_cache()
        self.rebuild_changed_rows()
        self.sort_field = None
        self.sort_reverse = False
        self.original_order = list(range(UNIT_SLOT_COUNT))
        self.view_order = self.original_order[:]
        self.rebuild_view_lookup()
        self.set_selected_slots([])
        self.render_table()

    def try_autoload_bin(self) -> None:
        candidates = find_candidate_bins()
        if candidates:
            self.load_bin(candidates[0], update_status=False)
            self.set_status(f"Auto-loaded {os.path.basename(candidates[0])}.", GOOD)
        else:
            self.render_table()

    def open_bin_file(self) -> None:
        path = filedialog.askopenfilename(title="Open LINKFILE_000.bin", initialdir=os.path.dirname(self.current_bin_path) if self.current_bin_path else SCRIPT_DIR, filetypes=[("BIN files", "*.bin *.BIN"), ("All files", "*.*")])
        if path:
            self.load_bin(path)

    def load_bin(self, path: str, *, update_status: bool = True) -> None:
        try:
            block = read_unit_block_from_bin(path)
            backup, backup_path, created = self.ensure_backup_for_bin(path, block)
        except Exception as exc:
            messagebox.showerror("Open BIN Failed", str(exc))
            self.set_status("Could not load BIN.", BAD)
            return
        self.current_bin_path = path
        self.backup_block = backup
        self.backup_path = backup_path
        self.load_block_into_memory(block, source_path=None, compare_block=backup)
        if update_status:
            suffix = " Backup captured." if created else " Backup already available."
            self.set_status(f"Loaded {os.path.basename(path)}.{suffix}", GOOD)

    def open_mod_file(self) -> None:
        path = filedialog.askopenfilename(title="Open Saved WO3 Unit Block", initialdir=os.path.dirname(self.current_mod_path) if self.current_mod_path else MOD_DIR, filetypes=[("WO3 unit block", "*.bin *.BIN"), ("All files", "*.*")])
        if path:
            self.load_mod_path(path)

    def load_mod_path(self, path: str, *, update_status: bool = True) -> None:
        try:
            block = read_mod_block(path)
        except Exception as exc:
            messagebox.showerror("Open Mod Failed", str(exc))
            self.set_status("Could not load mod.", BAD)
            return
        compare = self.backup_block if len(self.backup_block) == UNIT_BLOCK_SIZE else block
        self.load_block_into_memory(block, source_path=path, compare_block=compare)
        if update_status:
            self.set_status(f"Loaded {os.path.basename(path)} into memory.", GOOD)

    def refresh_memory_from_source(self) -> None:
        if self.current_bin_path:
            self.load_bin(self.current_bin_path, update_status=False)
            self.set_status(f"Refreshed memory from {os.path.basename(self.current_bin_path)}.", GOOD)
            return
        if self.current_mod_path:
            self.load_mod_path(self.current_mod_path, update_status=False)
            self.set_status(f"Refreshed memory from {os.path.basename(self.current_mod_path)}.", GOOD)
            return
        messagebox.showinfo("Nothing To Refresh", "Open a BIN or load a mod file first.")
        self.set_status("Open a BIN or load a mod file before refreshing.", WARN)

    def save_current_mod_as(self) -> None:
        if not self.working_block_loaded():
            messagebox.showinfo("Nothing To Save", "Load a BIN or a mod file first.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(title="Save Current Unit Block As", initialdir=MOD_DIR, initialfile=f"wo3_unit_mod_{stamp}.bin", defaultextension=".bin", filetypes=[("WO3 unit block", "*.bin"), ("All files", "*.*")])
        if not path:
            return
        try:
            write_mod_block(path, self.current_block())
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))
            self.set_status("Could not save mod.", BAD)
            return
        self.current_mod_path = path
        self.working_origin_block = self.current_block()
        self.rebuild_changed_rows()
        self.set_status(f"Saved {os.path.basename(path)}.", GOOD)

    def apply_current_to_bin(self) -> None:
        if not self.current_bin_path:
            messagebox.showinfo("No BIN Loaded", "Open LINKFILE_000.bin first.")
            return
        if not self.working_block_loaded():
            messagebox.showinfo("Nothing To Apply", "Load a BIN or a mod file first.")
            return
        try:
            write_unit_block_to_bin(self.current_bin_path, self.current_block())
        except Exception as exc:
            messagebox.showerror("Apply Failed", str(exc))
            self.set_status("Could not apply to BIN.", BAD)
            return
        self.set_status(f"Applied current memory to {os.path.basename(self.current_bin_path)}.", GOOD)

    def restore_bin_from_backup(self) -> None:
        if not self.current_bin_path or not self.backup_path or len(self.backup_block) != UNIT_BLOCK_SIZE:
            messagebox.showinfo("No Backup Ready", "Open LINKFILE_000.bin first so a backup can be created.")
            return
        if not messagebox.askyesno("Restore Original Unit Block?", f"This will restore the backed-up unit block to:\n{self.current_bin_path}\n\nContinue?"):
            return
        try:
            write_unit_block_to_bin(self.current_bin_path, self.backup_block)
        except Exception as exc:
            messagebox.showerror("Restore Failed", str(exc))
            self.set_status("Could not restore BIN.", BAD)
            return
        self.set_status(f"Restored {os.path.basename(self.current_bin_path)} from backup.", GOOD)

    def select_range_from_entry(self) -> None:
        try:
            slots = parse_slot_expression(self.range_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid Slot Expression", str(exc))
            self.set_status("Slot expression could not be parsed.", BAD)
            return
        self.set_selected_slots(slots, primary_slot=slots[-1] if slots else None)
        self.set_status(f"Selected {len(slots)} slot{'s' if len(slots) != 1 else ''}.", GOOD)

    def on_close_request(self) -> None:
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
    messagebox.showerror(APP_TITLE, f"A fatal error occurred.\n\nA crash log was written to:\n{CRASH_LOG_PATH}\n\n{exc}")
    root.destroy()


def main() -> None:
    app = WO3SteelEditor()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report_fatal_error(exc)
