import os, struct, threading, queue, tkinter as tk
from tkinter import ttk, filedialog, messagebox

# G1L/KOVS constants

G1L_SIG = bytes.fromhex("5F 4C 31 47 30 30 30 30")  # _L1G0000
KOVS_MAGIC = b"KOVS"
KTSS_MAGIC = b"KTSS"
KVS_EXT = ".kvs"
KTSS_EXT = ".ktss"


class G1LError(Exception):
    """Raised for G1L parsing/packing/unpacking errors"""

# GUI coloring

def setup_lilac_styles():
    style = ttk.Style()
    try:
        style.theme_use("clam")  # clam respects bg colors nicely
    except tk.TclError:
        pass
    style.configure("Lilac.TFrame",  background=LILAC)
    style.configure("Lilac.TLabel",  background=LILAC, foreground="black", padding=0)
    style.map("Lilac.TLabel", background=[("active", LILAC)])
    
# Binary helpers

def read_u32le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def write_u32le_into(buf: bytearray, off: int, val: int) -> None:
    struct.pack_into("<I", buf, off, val)

def parse_g1l_header(data: bytes) -> dict:
    """
    G1L format (from Orochi 3 testing) :

      0x00  8   signature: _L1G0000
      0x08  u32 file_size (absolute size of entire G1L file)
      0x0C  u32 meta_size (absolute size of G1L metadata starting at 0)
      0x10  u32 unknown
      0x14  u32 toc_count (entry count)
      0x18  ... TOC: toc_count entries, each 4 bytes: u32 absolute offset to KOVS
            (metadata may contain additional bytes/padding after the TOC, meta_size covers all metadata)

    KOVS block at each TOC offset:
      0x00 4  'KOVS'
      0x04 u32 payload_size
      0x08 0x18 bytes metadata
      0x20 payload bytes (payload_size)
      Total extracted KVS bytes = 0x20 + payload_size  (must be preserved as a whole)
    """
    if len(data) < 0x18:
        raise G1LError("File too small to be a valid G1L.")

    sig = data[0:8]
    if sig != G1L_SIG:
        raise G1LError(f"Bad G1L signature. Expected {G1L_SIG.hex(' ')}, got {sig.hex(' ')}")

    file_size_field = read_u32le(data, 0x08)
    meta_size = read_u32le(data, 0x0C)
    unk1 = read_u32le(data, 0x10)
    toc_count = read_u32le(data, 0x14)

    actual_size = len(data)

    if meta_size < 0x18:
        raise G1LError(f"Metadata size too small: 0x{meta_size:X}")
    if meta_size > actual_size:
        raise G1LError(f"Metadata size exceeds file length: meta=0x{meta_size:X}, file=0x{actual_size:X}")

    toc_min_end = 0x18 + toc_count * 4
    if toc_min_end > meta_size:
        raise G1LError(
            f"Metadata too small for TOC: meta_size=0x{meta_size:X}, "
            f"needs at least 0x{toc_min_end:X} for toc_count={toc_count}"
        )

    return {
        "sig": sig,
        "file_size_field": file_size_field,
        "meta_size": meta_size,
        "unk1": unk1,
        "toc_count": toc_count,
        "actual_size": actual_size,
        "toc_min_end": toc_min_end,
    }


def read_g1l_toc_offsets(data: bytes, toc_count: int) -> list[int]:
    """Returns list of absolute offsets (u32) parsed from TOC"""
    toc_off = 0x18
    offsets = []
    for i in range(toc_count):
        off = read_u32le(data, toc_off + i * 4)
        offsets.append(off)
    return offsets


def read_embedded_block(data: bytes, block_off: int) -> tuple[bytes, bytes]:
    """
    Reads and returns (block_bytes, magic) for a supported embedded block at block_off

    Supported blocks:

    KOVS:
      +0x00 4  'KOVS'
      +0x04 u32 payload_size
      +0x08 0x18 bytes metadata
      +0x20 payload bytes (payload_size)
      Total size = 0x20 + payload_size

    KTSS:
      +0x00 4  'KTSS'
      +0x04 u32 total_size (absolute size of this KTSS block, including header/size)
      +0x08 payload bytes (total_size - 8)
      Total size = total_size
    """
    if block_off < 0 or block_off + 8 > len(data):
        raise G1LError(f"Block offset out of bounds: 0x{block_off:X}")

    magic = data[block_off:block_off + 4]

    if magic == KOVS_MAGIC:
        if block_off + 0x20 > len(data):
            raise G1LError(f"KOVS header out of bounds at 0x{block_off:X}")
        payload_size = read_u32le(data, block_off + 4)
        total_size = 0x20 + payload_size
    elif magic == KTSS_MAGIC:
        total_size = read_u32le(data, block_off + 4)
        if total_size < 8:
            raise G1LError(f"KTSS size too small at 0x{block_off:X}: 0x{total_size:X}")
    else:
        raise G1LError(f"Unsupported block magic at 0x{block_off:X}: {magic!r}")

    end = block_off + total_size
    if end > len(data):
        raise G1LError(
            f"Block exceeds file: off=0x{block_off:X}, total=0x{total_size:X}, file=0x{len(data):X}"
        )

    return data[block_off:end], magic


def normalize_embedded_bytes(blob: bytes) -> bytes:
    """
    Ensures internal size fields match the actual file length for supported embedded blocks

    KOVS: rewrites payload_size at +0x04 to len(blob) - 0x20
    KTSS: rewrites total_size at +0x04 to len(blob)
    """
    if len(blob) < 8:
        raise G1LError("Embedded file too small (< 8).")

    magic = blob[0:4]
    b = bytearray(blob)

    if magic == KOVS_MAGIC:
        if len(blob) < 0x20:
            raise G1LError("KOVS file too small (< 0x20).")
        payload_len = len(blob) - 0x20
        cur = read_u32le(blob, 4)
        if cur != payload_len:
            write_u32le_into(b, 4, payload_len)
        return bytes(b)

    if magic == KTSS_MAGIC:
        cur = read_u32le(blob, 4)
        if cur != len(blob):
            write_u32le_into(b, 4, len(blob))
        return bytes(b)

    raise G1LError(f"Unsupported embedded file magic: {magic!r}")

# File helpers

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def is_g1l_path(p: str) -> bool:
    return p.lower().endswith(".g1l")

def is_payload_path(p: str) -> bool:
    return p.lower().endswith((KVS_EXT, KTSS_EXT))

def stem_is_5digits_numbered(filename: str) -> bool:
    base, ext = os.path.splitext(filename)
    return base.isdigit() and len(base) == 5 and ext.lower() in (KVS_EXT, KTSS_EXT)

def kvs_sort_key(path: str) -> int:
    """Natural numeric sort key for 00000.kvs style names"""
    name = os.path.basename(path)
    base, _ = os.path.splitext(name)
    if base.isdigit():
        return int(base)
    return 10**18

def collect_files_in_folder(folder: str, recursive: bool):
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for fn in filenames:
                yield os.path.join(dirpath, fn)
    else:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                yield p

# Core operations

def unpack_g1l(
    g1l_path: str,
    log_cb=None,
    progress_cb=None,   # expects (done_entries, total_entries)
    status_cb=None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, int]:
    """
    Unpacks a single G1L into a sibling folder named after the G1L base name
    Writes 5 digit numbered .kvs files in TOC order

    TOC entries are offsets only, each points to a KOVS block
    Each extracted .kvs is the full 0x20 header+metadata plus payload bytes
    """
    if log_cb:
        log_cb(f"Reading: {g1l_path}")

    with open(g1l_path, "rb") as f:
        data = f.read()

    h = parse_g1l_header(data)
    offsets = read_g1l_toc_offsets(data, h["toc_count"])

    base = os.path.splitext(os.path.basename(g1l_path))[0]
    out_dir = os.path.join(os.path.dirname(g1l_path), base)
    safe_makedirs(out_dir)

    if log_cb:
        log_cb(f"Meta size: 0x{h['meta_size']:X} | TOC count: {h['toc_count']} | Out: {out_dir}")

    total = len(offsets)
    for i, kovs_off in enumerate(offsets, start=1):
        if cancel_event and cancel_event.is_set():
            raise G1LError("Cancelled by user.")

        blob, magic = read_embedded_block(data, kovs_off)
        kvs_bytes = blob

        ext = KVS_EXT if magic == KOVS_MAGIC else KTSS_EXT
        out_name = f"{i-1:05d}{ext}"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "wb") as wf:
            wf.write(kvs_bytes)

        # continuous log updates; progress/status every 10 entries (or first/last)
        if log_cb:
            log_cb(f"Wrote {out_name} (0x{len(kvs_bytes):X} bytes @ 0x{kovs_off:X})")

        if progress_cb and (i == 1 or i == total or i % 10 == 0):
            progress_cb(i, total)
        if status_cb and (i == 1 or i == total or i % 10 == 0):
            status_cb(f"Unpack {base}: {i}/{total}")

    if progress_cb:
        progress_cb(total, total)
    if status_cb:
        status_cb(f"Unpack {base}: {total}/{total}")

    return out_dir, total

def repack_g1l(
    original_g1l_path: str,
    kvs_folder: str,
    recursive: bool,
    log_cb=None,
    progress_cb=None,   # expects (done_files, total_files)
    status_cb=None,
    cancel_event: threading.Event | None = None,
) -> str:
    """
    Rebuilds a G1L by:
      copying original metadata bytes [0:meta_size] to preserve any padding/extra metadata
      overwriting TOC offsets at 0x18 with new offsets
      appending numbered .kvs files sequentially starting at meta_size
      patching final file size at 0x08

    Constraints:
      number of .kvs files must match original toc_count
      .kvs files must be named 00000.kvs, 00001.kvs, etc and are written in natural numeric order
      each .kvs must begin with KOVS and keep the full 0x20 header/payload
    """
    if log_cb:
        log_cb(f"Original: {original_g1l_path}")
        log_cb(f"KVS folder: {kvs_folder} | Recursive search: {recursive}")

    with open(original_g1l_path, "rb") as f:
        orig = f.read()

    h = parse_g1l_header(orig)
    expected_count = h["toc_count"]

    all_paths = [p for p in collect_files_in_folder(kvs_folder, recursive) if is_payload_path(p)]
    if not all_paths:
        raise G1LError("No numbered .kvs/.ktss files found in the selected folder.")

    bad = [os.path.basename(p) for p in all_paths if not stem_is_5digits_numbered(os.path.basename(p))]
    if bad:
        raise G1LError(
            "These .kvs files do not follow the required 5 digit naming (00000.kvs):\n"
            + "\n".join(bad[:25])
            + ("\n... (more)" if len(bad) > 25 else "")
        )

    all_paths.sort(key=kvs_sort_key)

    if len(all_paths) != expected_count:
        raise G1LError(
            f"Entry count mismatch. Expected {expected_count} files (from original TOC count at 0x14), "
            f"found {len(all_paths)}."
        )

    base = os.path.splitext(os.path.basename(original_g1l_path))[0]
    out_path = os.path.join(os.path.dirname(original_g1l_path), f"{base}_repacked.g1l")

    # Copy original metadata (preserves anything beyond TOC too)
    meta_size = h["meta_size"]
    meta = bytearray(orig[:meta_size])

    # Sanity: ensure signature correct
    meta[0:8] = G1L_SIG

    # file size placeholder (patched at end)
    write_u32le_into(meta, 0x08, 0)

    # Make sure unknown and toc_count are preserved (in case the user selected a different original)
    write_u32le_into(meta, 0x10, h["unk1"])
    write_u32le_into(meta, 0x14, expected_count)

    # Zero TOC offsets area
    toc_off = 0x18
    for i in range(expected_count):
        write_u32le_into(meta, toc_off + i * 4, 0)

    # Append payloads sequentially starting at meta_size
    cur_off = meta_size
    new_offsets: list[int] = []

    total = len(all_paths)

    with open(out_path, "wb") as out:
        out.write(meta)

        for i, kvs_path in enumerate(all_paths, start=1):
            if cancel_event and cancel_event.is_set():
                raise G1LError("Cancelled by user.")

            with open(kvs_path, "rb") as f:
                kvs_bytes = f.read()

            kvs_bytes = normalize_embedded_bytes(kvs_bytes)

            new_offsets.append(cur_off)
            out.write(kvs_bytes)
            cur_off += len(kvs_bytes)

            # log continuously, progress/status every 10 files
            if log_cb:
                log_cb(f"Appended {os.path.basename(kvs_path)} (0x{len(kvs_bytes):X} bytes) -> off 0x{new_offsets[-1]:X}")

            if progress_cb and (i == 1 or i == total or i % 10 == 0):
                progress_cb(i, total)
            if status_cb and (i == 1 or i == total or i % 10 == 0):
                status_cb(f"Repack {base}: {i}/{total}")

    final_size = cur_off

    # Patch file size + TOC offsets in-place
    with open(out_path, "r+b") as out:
        out.seek(0x08)
        out.write(struct.pack("<I", final_size))

        out.seek(0x18)
        for off in new_offsets:
            out.write(struct.pack("<I", off))

    if progress_cb:
        progress_cb(total, total)
    if status_cb:
        status_cb("Repack done.")

    if log_cb:
        log_cb(f"Repacked written: {out_path}")
        log_cb(f"Final size: 0x{final_size:X}")

    return out_path

# Tkinter GUI (threaded)

class BaseTab:
    """Common UI plumbing: thread, cancel, log, progress"""

    def __init__(self, parent, tab_title: str):
        self.parent = parent
        self.tab_title = tab_title

        self.frame = ttk.Frame(parent)

        self._thread = None
        self._cancel_event = threading.Event()
        self._ui_queue = queue.Queue()

        self.status_var = tk.StringVar(value="Ready.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_common_bottom()
        self._poll_ui_queue()

    def _build_common_bottom(self):
        btns = ttk.Frame(self.frame)
        btns.pack(fill="x", padx=10, pady=(10, 0))

        self.start_btn = ttk.Button(btns, text=f"Start {self.tab_title}", command=self.start)
        self.start_btn.pack(side="left")

        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))

        prog = ttk.Frame(self.frame)
        prog.pack(fill="x", padx=10, pady=(10, 0))

        self.progressbar = ttk.Progressbar(
            prog, orient="horizontal", mode="determinate", variable=self.progress_var, maximum=100.0
        )
        self.progressbar.pack(fill="x")

        ttk.Label(self.frame, textvariable=self.status_var).pack(fill="x", padx=10, pady=(6, 0))

        log_frame = ttk.LabelFrame(self.frame, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, height=16, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _set_controls_running(self, running: bool):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.cancel_btn.configure(state="normal" if running else "disabled")

    # Thread safe UI updates
    def log(self, msg: str):
        self._ui_queue.put(("log", msg))

    def set_status(self, msg: str):
        self._ui_queue.put(("status", msg))

    def set_progress(self, pct: float):
        self._ui_queue.put(("progress", pct))

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def cancel(self):
        if self._thread and self._thread.is_alive():
            self._cancel_event.set()
            self.set_status("Cancel requested")

    def _poll_ui_queue(self):
        try:
            # Drain multiple messages per tick
            for _ in range(200):
                kind, payload = self._ui_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    self.progress_var.set(float(payload))
                elif kind == "error":
                    self._append_log(f"ERROR: {payload}")
                    self.status_var.set("Error.")
                    messagebox.showerror(f"{self.tab_title} error", payload)
                    self._set_controls_running(False)
                elif kind == "done":
                    self._set_controls_running(False)
        except queue.Empty:
            pass

        self.frame.after(50, self._poll_ui_queue)

    def start(self):
        raise NotImplementedError


class UnpackTab(BaseTab):
    """
    Batch unpack:
      select a folder
      optional recursive scanning
      finds .g1l files and unpacks each into <g1l_basename>/00000.kvs, etc
    """

    def __init__(self, parent):
        super().__init__(parent, "Unpack")

        self.folder_var = tk.StringVar(value="")
        self.recursive_var = tk.BooleanVar(value=True)

        self._build_inputs()

    def _build_inputs(self):
        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Input folder:").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(top, textvariable=self.folder_var, width=65)
        entry.grid(row=0, column=1, padx=(8, 8), sticky="we")
        browse_btn = ttk.Button(top, text="Select Folder", command=self._select_folder)
        browse_btn.grid(row=0, column=2, sticky="e")

        recursive_chk = ttk.Checkbutton(top, text="Include subdirectories", variable=self.recursive_var)
        recursive_chk.grid(row=1, column=1, sticky="w", pady=(8, 0))

        hint = ttk.Label(
            top,
            text="Unpacks every .g1l found into a sibling folder named after the G1L (no extension).",
        )
        hint.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        top.columnconfigure(1, weight=1)

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing .g1l files")
        if folder:
            self.folder_var.set(folder)

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Invalid folder", "Please select a valid input folder.")
            return

        self._cancel_event.clear()
        self.progress_var.set(0.0)
        self.status_var.set("Starting...")
        self._set_controls_running(True)
        self._clear_log()

        self._thread = threading.Thread(
            target=self._run_worker,
            args=(folder, self.recursive_var.get()),
            daemon=True,
        )
        self._thread.start()

    def _run_worker(self, folder: str, recursive: bool):
        try:
            self.log(f"[Unpack] Folder: {folder}")
            self.log(f"[Unpack] Recursive: {recursive}")
            self.log("Scanning for .g1l files")

            g1l_files = [p for p in collect_files_in_folder(folder, recursive) if is_g1l_path(p)]
            g1l_files.sort(key=lambda p: p.lower())

            total_g1l = len(g1l_files)
            if total_g1l == 0:
                self.log("No .g1l files found.")
                self.set_status("Done (no .g1l files).")
                self.set_progress(100.0)
                self._ui_queue.put(("done", None))
                return

            self.log(f"Found {total_g1l} .g1l file(s).")
            self.set_status(f"Unpacking {total_g1l} G1L file(s)")

            for file_idx, g1l_path in enumerate(g1l_files, start=1):
                if self._cancel_event.is_set():
                    self.log("Cancelled by user.")
                    self.set_status("Cancelled.")
                    self._ui_queue.put(("done", None))
                    return

                base = os.path.splitext(os.path.basename(g1l_path))[0]

                def _progress_cb(done_entries: int, total_entries: int, _file_idx=file_idx):
                    if total_entries <= 0:
                        pct = (_file_idx / total_g1l) * 100.0
                    else:
                        pct = ((_file_idx - 1) + (done_entries / total_entries)) / total_g1l * 100.0
                    self.set_progress(pct)

                def _status_cb(msg: str):
                    self.set_status(msg)

                out_dir, entry_total = unpack_g1l(
                    g1l_path,
                    log_cb=self.log,
                    progress_cb=_progress_cb,
                    status_cb=_status_cb,
                    cancel_event=self._cancel_event,
                )
                self.log(f"OK -> {out_dir} ({entry_total} entries)")

            self.set_progress(100.0)
            self.set_status("Done.")
            self.log("All done.")
            self._ui_queue.put(("done", None))

        except Exception as e:
            if "cancelled" in str(e).lower():
                self.log("Cancelled by user.")
                self.set_status("Cancelled.")
                self._ui_queue.put(("done", None))
                return
            self._ui_queue.put(("error", str(e)))

class RepackTab(BaseTab):
    """
    Single repack flow:
      select original .g1l
      select folder containing numbered 00000.kvs, 00001.kvs, etc
      optional recursive search for those .kvs
      writes <basename>_repacked.g1l next to original
    """

    def __init__(self, parent):
        super().__init__(parent, "Repack")

        self.g1l_var = tk.StringVar(value="")
        self.folder_var = tk.StringVar(value="")
        self.recursive_var = tk.BooleanVar(value=False)

        self._build_inputs()

    def _build_inputs(self):
        top = ttk.Frame(self.frame)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(top, text="Original G1L file:").grid(row=0, column=0, sticky="w")
        g1l_entry = ttk.Entry(top, textvariable=self.g1l_var, width=65)
        g1l_entry.grid(row=0, column=1, padx=(8, 8), sticky="we")
        g1l_btn = ttk.Button(top, text="Select File", command=self._select_g1l)
        g1l_btn.grid(row=0, column=2, sticky="e")

        ttk.Label(top, text="Unpacked KVS folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        folder_entry = ttk.Entry(top, textvariable=self.folder_var, width=65)
        folder_entry.grid(row=1, column=1, padx=(8, 8), sticky="we", pady=(8, 0))
        folder_btn = ttk.Button(top, text="Select Folder", command=self._select_folder)
        folder_btn.grid(row=1, column=2, sticky="e", pady=(8, 0))

        recursive_chk = ttk.Checkbutton(top, text="Include subdirectories", variable=self.recursive_var)
        recursive_chk.grid(row=2, column=1, sticky="w", pady=(8, 0))

        hint = ttk.Label(
            top,
            text="Folder must contain 5 digit numbered files like 00000.kvs/00000.ktss (must match TOC count at 0x14).",
        )
        hint.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        top.columnconfigure(1, weight=1)

    def _select_g1l(self):
        p = filedialog.askopenfilename(
            title="Select original .g1l file",
            filetypes=[("G1L files", "*.g1l"), ("All files", "*.*")],
        )
        if p:
            self.g1l_var.set(p)
            base = os.path.splitext(os.path.basename(p))[0]
            auto = os.path.join(os.path.dirname(p), base)
            if os.path.isdir(auto) and not self.folder_var.get().strip():
                self.folder_var.set(auto)

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing unpacked numbered files")
        if folder:
            self.folder_var.set(folder)

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        g1l_path = self.g1l_var.get().strip()
        folder = self.folder_var.get().strip()

        if not g1l_path or not os.path.isfile(g1l_path) or not is_g1l_path(g1l_path):
            messagebox.showerror("Invalid G1L", "Please select a valid .g1l file.")
            return
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Invalid folder", "Please select the folder containing unpacked numbered files.")
            return

        self._cancel_event.clear()
        self.progress_var.set(0.0)
        self.status_var.set("Starting...")
        self._set_controls_running(True)
        self._clear_log()

        self._thread = threading.Thread(
            target=self._run_worker,
            args=(g1l_path, folder, self.recursive_var.get()),
            daemon=True,
        )
        self._thread.start()

    def _run_worker(self, g1l_path: str, folder: str, recursive: bool):
        try:
            self.log("[Repack] Starting")

            def _progress_cb(done_files: int, total_files: int):
                pct = (done_files / max(total_files, 1)) * 100.0
                self.set_progress(pct)

            def _status_cb(msg: str):
                self.set_status(msg)

            out = repack_g1l(
                g1l_path,
                folder,
                recursive,
                log_cb=self.log,
                progress_cb=_progress_cb,
                status_cb=_status_cb,
                cancel_event=self._cancel_event,
            )

            self.set_progress(100.0)
            self.set_status("Done.")
            self.log(f"Done -> {out}")
            self._ui_queue.put(("done", None))

        except Exception as e:
            if "cancelled" in str(e).lower():
                self.log("Cancelled by user.")
                self.set_status("Cancelled.")
                self._ui_queue.put(("done", None))
                return
            self._ui_queue.put(("error", str(e)))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wild Liberd, G1L Tool")
        self.geometry("920x600")
        self.minsize(840, 520)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.unpack_tab = UnpackTab(notebook)
        notebook.add(self.unpack_tab.frame, text="Unpack")

        self.repack_tab = RepackTab(notebook)
        notebook.add(self.repack_tab.frame, text="Repack")

        self._build_menu()

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

if __name__ == "__main__":
    App().mainloop()
