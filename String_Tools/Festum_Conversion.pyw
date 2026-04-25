import os, json, struct, threading, queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Festum Conversion, Powered By Mir
# Handles XL19, ECB, StringTable, legacy XL01, XL02, EM/MESC formats

APP_TITLE = "Festum Conversion, Binary Text Translator"
DEFAULT_ENCODING = "utf-8"

MOD_TAIL_SIZE = 6  # Aldnoah Engine taildata length

SUPPORTED_FILE_PATTERNS = "*.xl *.em *.mesc *.ecb *.bin *.dat *.tbl"

KOEI_EN_TEXT_REPLACEMENTS = {
    "Ā": "A",
    "ā": "a",
    "Ē": "E",
    "ē": "e",
    "Ī": "I",
    "ī": "i",
    "Ō": "O",
    "ō": "o",
    "Ū": "U",
    "ū": "u",
    "\xdb": "o",
    "\xdc": "u",
}

# Helpers

def koei_encode(s: str) -> bytes:
    return encode_text(s, "koei-en")

def normalize_koei_en_text(s: str) -> str:
    for src, dst in KOEI_EN_TEXT_REPLACEMENTS.items():
        s = s.replace(src, dst)
    return s

def decode_text(raw: bytes, encoding: str, *, strict: bool = False) -> str:
    if encoding.lower() == "koei-en":
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            if strict:
                text = raw.decode("latin-1", errors="strict")
            else:
                text = raw.decode("latin-1", errors="replace")
        return normalize_koei_en_text(text)

    if strict:
        return raw.decode(encoding, errors="strict")

    try:
        return raw.decode(encoding, errors="strict")
    except Exception:
        return raw.decode(encoding, errors="replace")

def encode_text(s: str, encoding: str) -> bytes:
    if encoding.lower() == "koei-en":
        return normalize_koei_en_text(s).encode("utf-8", errors="strict")
    return s.encode(encoding, errors="strict")

def align(value: int, a: int) -> int:
    return (value + (a - 1)) & ~(a - 1)

def read_u16_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]

def read_i16_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<h", b, off)[0]

def read_u32_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def read_i32_le(b: bytes, off: int) -> int:
    return struct.unpack_from("<i", b, off)[0]

def write_u16_le(buf: bytearray, off: int, val: int) -> None:
    struct.pack_into("<H", buf, off, val & 0xFFFF)

def write_u32_le(buf: bytearray, off: int, val: int) -> None:
    struct.pack_into("<I", buf, off, val & 0xFFFFFFFF)

def write_i32_le(buf: bytearray, off: int, val: int) -> None:
    struct.pack_into("<i", buf, off, int(val))

def read_c_string(data: bytes, off: int, encoding: str) -> str:
    if off < 0 or off >= len(data):
        return ""
    end = data.find(b"\x00", off)
    if end == -1:
        end = len(data)
    raw = data[off:end]
    return decode_text(raw, encoding)

def find_c_string_end(data: bytes, off: int) -> int:
    if off < 0 or off >= len(data):
        return off
    end = data.find(b"\x00", off)
    if end == -1:
        return len(data)
    return end + 1  # include null

def safe_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default

def split_mod_taildata(blob: bytes) -> tuple[bytes, bytes]:
    """
    Aldnoah Engine appends a 6 byte taildata to extracted files for mod manager bookkeeping
    This taildata is not part of the underlying file format so we strip it for parsing/writing
    then re-append it as the last 6 bytes on save
    If the file is too small, returns (blob, b'')
    """
    if blob is None:
        return b"", b""
    if len(blob) < MOD_TAIL_SIZE:
        return blob, b""
    return blob[:-MOD_TAIL_SIZE], blob[-MOD_TAIL_SIZE:]

# Format Parsing/Writing

# XL19 types
XLTYPE_STRING_PTR = 0x00
XLTYPE_INT32 = 0x01
XLTYPE_INT16 = 0x02
XLTYPE_INT8 = 0x03
XLTYPE_UINT32 = 0x04
XLTYPE_UINT16 = 0x05
XLTYPE_UINT8 = 0x06
XLTYPE_SINGLE = 0x07
XLTYPE_NOP = 0xFF

def xl19_type_size(t: int) -> int:
    if t in (XLTYPE_INT32, XLTYPE_UINT32, XLTYPE_SINGLE, XLTYPE_STRING_PTR):
        return 4
    if t in (XLTYPE_INT16, XLTYPE_UINT16):
        return 2
    if t in (XLTYPE_INT8, XLTYPE_UINT8):
        return 1
    if t == XLTYPE_NOP:
        return 0
    # Unknown types, assume 0 so we don't walk off but this will likely fail validation later
    return 0

def detect_format(path: str, data: bytes):
    """Return a string format key"""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".em":
        return "em"
    if ext == ".mesc":
        return "mesc"

    # XL family, magic 0x4C58 ('XL') and version 0x13 (b'XL\\x13\\x00')
    if len(data) >= 4:
        magic = read_u16_le(data, 0)
        ver = read_u16_le(data, 2)
        if magic == 0x4C58:
            if ver != 0x13:
                return None

            # Try to interpret as XLHeader
            if len(data) >= 16:
                file_size_field = read_u16_le(data, 4)
                types = read_u16_le(data, 6)
                sets = read_u16_le(data, 8)
                width = read_i16_le(data, 10)
                table_off = read_i16_le(data, 12)

                # XL19, validate plausible table layout
                plausible = (
                    0 <= types <= 0x400 and
                    0 <= sets <= 0xFFFF and
                    0 < width < 0x4000 and
                    16 + types <= len(data) and
                    0 <= table_off < len(data) and
                    table_off + (sets * width) <= len(data)
                )
                # Older XL01/XL02 files can also start with XL\\x13\\x00 but don't follow XLHeader
                if plausible:
                    return "xl19"

            # Fallback, legacy XL subtypes
            return "xl_legacy"

    # ECB, header is 24 bytes, then entries, then dynamic blob
    if len(data) >= 24:
        field_count = read_i32_le(data, 4)
        entry_count = read_i32_le(data, 8)
        stride = read_i32_le(data, 12)
        dyn_ptr = read_i32_le(data, 16)
        total_size = read_i32_le(data, 20)
        header_ok = (
            0 <= field_count <= 0x2000 and
            0 <= entry_count <= 0x200000 and
            0 < stride <= 0x100000 and
            0 <= dyn_ptr <= len(data) and
            0 <= total_size <= len(data)
        )
        if header_ok:
            entries_off = align(24, 0x10)
            # Basic structural check, entries block shouldn't overlap dynamic blob
            if dyn_ptr >= entries_off and (entries_off + entry_count * stride) <= dyn_ptr and total_size >= dyn_ptr:
                # Also ensure we can read at least 1 ECBStringHeader (8 bytes) in dynamic area
                if (total_size - dyn_ptr) >= 8:
                    return "ecb"

    # StringTable, int32 count + count*(offset, size)
    if len(data) >= 12:
        count = read_i32_le(data, 0)
        if 0 < count < 0x200000:
            table_bytes = 4 + count * 8
            if table_bytes <= len(data):
                ok = True
                for i in range(min(count, 64)):
                    off = read_i32_le(data, 4 + i*8)
                    size = read_i32_le(data, 8 + i*8)
                    if off < 0 or off >= len(data):
                        ok = False
                        break
                    if size < 0 or size > len(data):
                        ok = False
                        break
                if ok:
                    return "stringtable"

    return None

def parse_xl19(data: bytes, encoding: str):
    magic = read_u16_le(data, 0)
    ver = read_u16_le(data, 2)
    if magic != 0x4C58 or ver != 0x13:
        raise ValueError("Not an XL19 stream")

    header = {
        "magic": magic,
        "version": ver,
        "file_size_field": read_u16_le(data, 4),
        "types_count": read_u16_le(data, 6),
        "sets": read_u16_le(data, 8),
        "width": read_i16_le(data, 10),
        "table_offset": read_i16_le(data, 12),
        "unknown": read_i16_le(data, 14),
    }
    types_count = header["types_count"]
    sets = header["sets"]
    width = header["width"]
    table_offset = header["table_offset"]

    if width <= 0:
        raise ValueError("XL19 width invalid")

    if 16 + types_count > len(data):
        raise ValueError("XL19 types out of range")

    types = list(data[16:16+types_count])

    end_of_table = table_offset + sets * width
    if table_offset < 0 or end_of_table > len(data):
        raise ValueError("XL19 table region out of range")

    string_entries = []
    max_string_end = end_of_table
    for row in range(sets):
        row_base = table_offset + row * width
        local = 0
        for col, t in enumerate(types):
            if t == XLTYPE_NOP:
                continue
            if t == XLTYPE_STRING_PTR:
                ptr_off = row_base + local
                if ptr_off + 4 > len(data):
                    break
                rel = read_i32_le(data, ptr_off)
                abs_off = table_offset + rel
                s = read_c_string(data, abs_off, encoding)
                string_entries.append({
                    "row": row,
                    "col": col,
                    "ptr_off": ptr_off,
                    "old_rel": rel,
                    "old_abs": abs_off,
                })
                max_string_end = max(max_string_end, find_c_string_end(data, abs_off))
                local += 4
            else:
                sz = xl19_type_size(t)
                local += sz
        # We don't hard fail on misalignment, some games may pad/pack oddly

    strings = []
    for ent in string_entries:
        strings.append(read_c_string(data, ent["old_abs"], encoding))

    meta = {
        "fmt": "xl19",
        "header": header,
        "types": types,
        "string_entries": string_entries,
        "end_of_table": end_of_table,
        "max_string_end": max_string_end,
        "encoding": encoding,
    }
    return strings, meta

def write_xl19(original: bytes, strings: list, meta: dict, encoding: str):
    """
    XL19 max safety writer
    
    Preserve the entire original file byte for byte
    Append the rebuilt translated string pool at the end of the file
    Redirect each string pointer (relative to table offset) to the newly appended string
    This avoids overwriting any unknown blobs/regions that might exist between the table and old strings
    """
    header = meta["header"]
    string_entries = meta["string_entries"]

    if len(strings) != len(string_entries):
        raise ValueError("String count mismatch for XL19")

    table_off = header["table_offset"]
    if table_off < 0 or table_off >= len(original):
        raise ValueError("XL19 table_offset invalid")

    out = bytearray(original)

    new_rels = []
    for s in strings:
        b = encode_text(s, encoding)

        abs_off = len(out)  # append to end for max safety
        rel = abs_off - table_off
        
        # signed int32 safety
        if not (-0x80000000 <= rel <= 0x7FFFFFFF):
            raise ValueError("XL19 string pool grew too large, relative offset out of int32 range")
        
        new_rels.append(rel)
        out.extend(b)
        out.append(0)

    # Patch pointers inplace
    for ent, rel in zip(string_entries, new_rels):
        write_i32_le(out, ent["ptr_off"], rel)

    # Update header's file size field to new file size if it fits
    # If it doesn't fit, leaving it unchanged is safer than overflowing
    new_size = len(out)
    if 0 <= new_size <= 0xFFFF:
        write_u16_le(out, 4, new_size)

    return bytes(out)

def parse_ecb(data: bytes, encoding: str):
    if len(data) < 24:
        raise ValueError("ECB too small")

    header = {
        "magic": read_u32_le(data, 0),
        "field_count": read_i32_le(data, 4),
        "entry_count": read_i32_le(data, 8),
        "stride": read_i32_le(data, 12),
        "dynamic_ptr": read_i32_le(data, 16),
        "total_size": read_i32_le(data, 20),
    }

    field_count = header["field_count"]
    entry_count = header["entry_count"]
    stride = header["stride"]
    dyn_ptr = header["dynamic_ptr"]
    total = header["total_size"]

    if total <= 0 or total > len(data):
        total = len(data)

    entries_off = align(24, 0x10)
    entries_end = entries_off + entry_count * stride
    if entries_end > dyn_ptr or dyn_ptr > total:
        raise ValueError("ECB layout invalid")

    dynamic = data[dyn_ptr:total]

    # Build a set of referenced dynamic offsets by scanning entry dwords
    candidates = {}
    for i in range(entry_count):
        base = entries_off + i * stride
        entry = data[base:base+stride]
        for p in range(0, stride - 3, 4):
            off = struct.unpack_from("<i", entry, p)[0]
            if off < 0 or off + 8 > len(dynamic):
                continue
            # ECBStringHeader, u32 hash, i16 charset, i16 size
            h = struct.unpack_from("<I", dynamic, off)[0]
            charset = struct.unpack_from("<h", dynamic, off+4)[0]
            size = struct.unpack_from("<h", dynamic, off+6)[0]
            if charset != 0x8:
                continue
            if size <= 1 or (off + 8 + (size - 1)) > len(dynamic):
                continue
            raw = dynamic[off+8:off+8+(size-1)]
            try:
                s = decode_text(raw, encoding, strict=True)
            except Exception:
                # not a valid utf8 skip
                continue
            # reject strings with too many control chars
            bad = sum(1 for ch in s if ord(ch) < 0x20 and ch not in "\t\r\n")
            if bad > 0:
                continue
            candidates[off] = {"hash": h, "charset": charset, "size": size, "text": s}

    # If we found nothing, do a small scan across dynamic as fallback
    if not candidates and len(dynamic) >= 8:
        for off in range(0, min(len(dynamic)-8, 0x20000)):  # cap scan
            charset = struct.unpack_from("<h", dynamic, off+4)[0]
            size = struct.unpack_from("<h", dynamic, off+6)[0]
            if charset != 0x8:
                continue
            if size <= 1 or (off + 8 + (size - 1)) > len(dynamic):
                continue
            raw = dynamic[off+8:off+8+(size-1)]
            try:
                s = decode_text(raw, encoding, strict=True)
            except Exception:
                continue
            candidates[off] = {"hash": struct.unpack_from("<I", dynamic, off)[0], "charset": charset, "size": size, "text": s}

    offsets = sorted(candidates.keys())
    strings = [candidates[o]["text"] for o in offsets]

    meta = {
        "fmt": "ecb",
        "header": header,
        "entries_off": entries_off,
        "dyn_ptr": dyn_ptr,
        "total": total,
        "stride": stride,
        "entry_count": entry_count,
        "offsets": offsets,          # in extraction order
        "offset_info": candidates,   # old_offset -> {hash, charset, size}
    }
    return strings, meta

def write_ecb(original: bytes, strings: list, meta: dict, encoding: str):
    offsets = meta["offsets"]
    info = meta["offset_info"]
    hdr = meta["header"]
    dyn_ptr = meta["dyn_ptr"]
    total = meta["total"]
    entries_off = meta["entries_off"]
    stride = meta["stride"]
    entry_count = meta["entry_count"]

    if len(strings) != len(offsets):
        raise ValueError("String count mismatch for ECB")

    # Preserve everything before dynamic blob exactly, header/padding/entries/any padding up to dyn_ptr
    prefix = bytearray(original[:dyn_ptr])

    # Preserve any tail after total and re-append after rebuilt dynamic
    tail = original[total:] if total <= len(original) else b""

    new_dynamic = bytearray()
    old_to_new = {}

    for off, s in zip(offsets, strings):
        # Preserve hash/charset, update size based on the selected text encoding
        h = info[off]["hash"]
        charset = 0x8
        b = encode_text(s, encoding)
        size = len(b) + 1
        new_off = len(new_dynamic)
        old_to_new[off] = new_off
        new_dynamic.extend(struct.pack("<Ihh", h, charset, size))
        new_dynamic.extend(b)
        new_dynamic.append(0)

    # Patch entry region, replace any int32 equal to an old string header offset with its new offset
    for i in range(entry_count):
        base = entries_off + i * stride
        if base + stride > len(prefix):
            break
        entry = bytearray(prefix[base:base+stride])
        for p in range(0, stride - 3, 4):
            val = struct.unpack_from("<i", entry, p)[0]
            if val in old_to_new:
                struct.pack_into("<i", entry, p, old_to_new[val])
        prefix[base:base+stride] = entry

    # Update size in header
    new_total = dyn_ptr + len(new_dynamic)
    if new_total < 0:
        new_total = dyn_ptr
    write_i32_le(prefix, 20, new_total)

    out = bytes(prefix) + bytes(new_dynamic) + tail
    return out

def parse_stringtable(data: bytes, encoding: str):
    count = read_i32_le(data, 0)
    if count <= 0:
        raise ValueError("StringTable count invalid")
    table_size = 4 + count * 8
    if table_size > len(data):
        raise ValueError("StringTable table out of range")

    records = []
    max_end = table_size
    for i in range(count):
        off = read_i32_le(data, 4 + i*8)
        size = read_i32_le(data, 8 + i*8)
        if off < 0 or off >= len(data):
            raise ValueError("StringTable offset out of range")
        s = read_c_string(data, off, encoding)
        records.append({"rec_off": 4 + i*8, "old_off": off, "old_size": size})
        max_end = max(max_end, find_c_string_end(data, off))

    strings = [read_c_string(data, r["old_off"], encoding) for r in records]
    meta = {"fmt": "stringtable", "count": count, "records": records, "table_size": table_size, "max_end": max_end}
    return strings, meta

def write_stringtable(original: bytes, strings: list, meta: dict, encoding: str):
    records = meta["records"]
    table_size = meta["table_size"]
    max_end = meta["max_end"]

    if len(strings) != len(records):
        raise ValueError("String count mismatch for StringTable")

    prefix = bytearray(original[:table_size])
    tail = original[max_end:] if max_end <= len(original) else b""

    out = bytearray(prefix)
    new_offsets = []
    for s in strings:
        b = encode_text(s, encoding)
        new_offsets.append(len(out))
        out.extend(b)
        out.append(0)

    # Patch offsets
    for r, off in zip(records, new_offsets):
        write_i32_le(out, r["rec_off"], off)

    out.extend(tail)
    return bytes(out)

# Legacy formats

TAIL_SIZE = MOD_TAIL_SIZE

def parse_xl_legacy(data: bytes, encoding: str):
    """
    XL01/XL02 support
    """
    if len(data) < 0x10:
        raise ValueError("XL legacy too small")
    sig = data[:4]
    if sig != b"XL\x13\x00":
        raise ValueError("Not XL legacy signature")
    # size field can be 2 bytes but some variants used 4
    size_val = read_u16_le(data, 4)
    size_field_len = 2
    size_field_offset = 4
    # If 2 byte size looks insane, try 4 byte at 4
    if size_val > len(data) or size_val < 0:
        if len(data) >= 8:
            size_val = read_u32_le(data, 4)
            size_field_len = 4

    subtype = read_u16_le(data, 6)
    count = read_u16_le(data, 8)
    toc_base = read_u32_le(data, 0x0C)

    tail_start = size_val
    if tail_start > len(data) or tail_start < 0:
        raise ValueError("Invalid size field/tail_start")

    # Preserve any bytes after tail_start inside the core file
    tail = data[tail_start:]

    # Read offsets
    offsets = []
    toc = data[toc_base:toc_base + (count * (4 if subtype == 1 else 5 if subtype == 2 else 0))]
    if subtype == 1:
        # 4 byte entries, 16 bit in bytes 2..3 is absolute offset
        for i in range(count):
            ent_off = toc_base + i*4
            if ent_off+4 > len(data): break
            ent = data[ent_off:ent_off+4]
            abs_off = struct.unpack("<H", ent[2:4])[0]
            offsets.append(abs_off)
    elif subtype == 2:
        # flag/u32 abs offset
        for i in range(count):
            ent_off = toc_base + i*5
            if ent_off+5 > len(data): break
            abs_off = read_u32_le(data, ent_off+1)
            offsets.append(abs_off)
    else:
        raise ValueError(f"Unsupported legacy XL subtype: {subtype}")

    strings = [read_c_string(data, off, encoding) for off in offsets]

    meta = {
        "fmt": "xl_legacy",
        "subtype": subtype,
        "count": count,
        "toc_base": toc_base,
        "offsets": offsets,
        "tail_start": tail_start,
        "tail": tail,
        "size_field_offset": size_field_offset,
        "size_field_len": size_field_len,
        "encoding": encoding,
    }
    return strings, meta

def write_xl_legacy(original: bytes, strings: list, meta: dict, encoding: str):
    subtype = meta["subtype"]
    count = meta["count"]
    toc_base = meta["toc_base"]
    tail = meta.get("tail", b"")
    size_field_offset = meta["size_field_offset"]
    size_field_len = meta["size_field_len"]

    if len(strings) != count:
        raise ValueError("Legacy XL string count mismatch")

    # Rebuild, keep everything up to the first string offset as prefix, then write strings, then re-append preserved tail bytes
    # The size field is treated as the start of this tail region
    # Use the minimum of offsets as the start of string block
    offsets = meta["offsets"]
    if not offsets:
        raise ValueError("No offsets in legacy XL meta")

    string_block_start = min(offsets)
    prefix = bytearray(original[:string_block_start])

    out = bytearray(prefix)
    new_offsets = []
    for s in strings:
        new_offsets.append(len(out))
        out.extend(encode_text(s, encoding))
        out.append(0)

    new_tail_start = len(out)
    out.extend(tail)

    # Patch size field to tail start
    if size_field_len == 2:
        write_u16_le(out, size_field_offset, new_tail_start)
    else:
        write_u32_le(out, size_field_offset, new_tail_start)

    # Patch TOC
    if subtype == 1:
        for i, off in enumerate(new_offsets):
            ent_off = toc_base + i*4
            if ent_off+4 > len(out): raise ValueError("Legacy XL TOC out of range")
            # Preserve first 2 bytes, rewrite u16 in last two bytes
            b0 = out[ent_off:ent_off+2]
            out[ent_off:ent_off+2] = b0
            out[ent_off+2:ent_off+4] = struct.pack("<H", off & 0xFFFF)
    elif subtype == 2:
        for i, off in enumerate(new_offsets):
            ent_off = toc_base + i*5
            if ent_off+5 > len(out): raise ValueError("Legacy XL TOC out of range")
            # preserve flag byte at ent_off
            # write u32 at ent_off+1
            struct.pack_into("<I", out, ent_off+1, off)
    else:
        raise ValueError("Unsupported legacy XL subtype in writer")

    return bytes(out)


def text_score(strings):
    """
    Rough heuristic used for how printable the combined text is
    0.0 -> garbage but if 1.0 -> all printable
    """
    combined = "\n".join(strings)
    if not combined:
        return 0.0
    printable = 0
    for ch in combined:
        if ch.isprintable() or ch in "\n\r\t":
            printable += 1
    return printable / max(1, len(combined))

def parse_em(data: bytes, encoding: str):
    # EM format
    # 0x00: 4 bytes signature
    # 0x04: 4 bytes size (tail_start)
    # 0x12: 2 bytes count
    # 0x18: TOC of count * 4 (direct offsets)
    if len(data) < 0x18:
        raise ValueError("EM too small")

    size_field = read_u32_le(data, 0x04)
    count = read_u16_le(data, 0x12)
    toc_start = 0x18
    toc_end = toc_start + count * 4

    if toc_end > len(data):
        raise ValueError("EM TOC out of range")

    tail_start = size_field
    if tail_start < toc_end or tail_start > len(data):
        raise ValueError("EM size/tail_start invalid")

    toc = [read_u32_le(data, toc_start + i*4) for i in range(count)]

    # Offsets point into the string region, duplicates can be valid when entries share text
    if not all(toc[i] >= toc_end and toc[i] < tail_start for i in range(count)):
        raise ValueError("EM TOC entries out of range")

    strings = []
    for i in range(count):
        start = toc[i]
        raw = data[start:tail_start].split(b"\x00", 1)[0]
        strings.append(decode_text(raw, encoding))

    if text_score(strings) < 0.4:
        raise ValueError("EM text score too low (likely not a text EM)")

    meta = {
        "fmt": "em",
        "count": count,
        "toc_start": toc_start,
        "size_field_offset": 0x04,
        "size_field_len": 4,
    }
    return strings, meta

def write_em(original: bytes, strings: list, meta: dict, encoding: str):
    """Append new strings at tail_start and move taildata, updating size/TOC offsets"""
    size_field_offset = meta["size_field_offset"]
    size_field_len = meta["size_field_len"]
    toc_start = meta["toc_start"]

    if size_field_len != 4:
        raise ValueError("EM expects 4 byte size field")
    if len(strings) != meta["count"]:
        raise ValueError("EM string count mismatch")

    tail_start = read_u32_le(original, size_field_offset)
    if tail_start < 0 or tail_start > len(original):
        raise ValueError("EM tail_start invalid")

    # Preserve any bytes after tail_start inside the core file
    tail = original[tail_start:]

    out = bytearray(original[:tail_start])
    new_offsets = []
    for s in strings:
        new_offsets.append(len(out))
        out.extend(encode_text(s, encoding))
        out.append(0)

    new_tail_start = len(out)
    out.extend(tail)

    # Update size field
    write_u32_le(out, size_field_offset, new_tail_start)

    # Update TOC
    for i, off in enumerate(new_offsets):
        write_u32_le(out, toc_start + i*4, off)

    return bytes(out)

def parse_mesc(data: bytes, encoding: str):
    # MESC format
    # 0x00: 4 bytes signature
    # 0x04: 4 bytes count
    # 0x08: 4 bytes size (tail_start)
    # 0x0C: TOC of count * 4 (direct offsets)
    if len(data) < 0x0C:
        raise ValueError("MESC too small")

    count = read_u32_le(data, 0x04)
    tail_start = read_u32_le(data, 0x08)
    toc_start = 0x0C
    toc_end = toc_start + count * 4

    if toc_end > len(data):
        raise ValueError("MESC TOC out of range")
    if tail_start < toc_end or tail_start > len(data):
        raise ValueError("MESC size/tail_start invalid")

    toc = [read_u32_le(data, toc_start + i*4) for i in range(count)]
    if not all(toc[i] >= toc_end and toc[i] < tail_start for i in range(count)):
        raise ValueError("MESC TOC entries out of range")

    strings = []
    for i in range(count):
        start = toc[i]
        raw = data[start:tail_start].split(b"\x00", 1)[0]
        strings.append(decode_text(raw, encoding))

    if text_score(strings) < 0.4:
        raise ValueError("MESC text score too low (likely not a text MESC)")

    meta = {
        "fmt": "mesc",
        "count": count,
        "toc_start": toc_start,
        "size_field_offset": 0x08,
        "size_field_len": 4,
    }
    return strings, meta

def write_mesc(original: bytes, strings: list, meta: dict, encoding: str):
    if len(strings) != meta["count"]:
        raise ValueError("MESC string count mismatch")

    tail_start = read_u32_le(original, meta["size_field_offset"])
    if tail_start < 0 or tail_start > len(original):
        raise ValueError("MESC tail_start invalid")
    tail = original[tail_start:]

    out = bytearray(original[:tail_start])
    new_offsets = []
    for s in strings:
        new_offsets.append(len(out))
        out.extend(encode_text(s, encoding))
        out.append(0)

    new_tail_start = len(out)
    out.extend(tail)

    write_u32_le(out, meta["size_field_offset"], new_tail_start)
    toc_start = meta["toc_start"]
    for i, off in enumerate(new_offsets):
        write_u32_le(out, toc_start + i*4, off)

    return bytes(out)

def apply_festum_gold_theme(root: tk.Tk) -> ttk.Style:
    """Gold/obsidian theme inspired by Festum halos and metallic silhouettes"""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    palette = {
        "bg": "#07111F",
        "panel": "#101827",
        "panel_alt": "#162033",
        "panel_glow": "#22304A",
        "gold": "#E6C66A",
        "gold_hot": "#FFE8A3",
        "gold_deep": "#A87925",
        "amber": "#C98B2C",
        "text": "#F8EBC4",
        "subtle": "#BBA56E",
        "muted": "#7D8AA3",
        "entry_bg": "#07101C",
        "tree_bg": "#0B1422",
        "tree_alt": "#101B2D",
        "select_bg": "#D9AD4E",
        "select_fg": "#08101C",
        "danger": "#E08C69",
        "border": "#3E4A61",
    }

    try:
        root.configure(bg=palette["bg"])
    except Exception:
        pass

    style.configure(".", background=palette["bg"], foreground=palette["text"], font=("Segoe UI", 10))
    style.configure("TFrame", background=palette["panel"])
    style.configure("App.TFrame", background=palette["bg"])
    style.configure("Hero.TFrame", background=palette["bg"])
    style.configure("Panel.TFrame", background=palette["panel"])
    style.configure("Toolbar.TFrame", background=palette["panel_alt"])

    style.configure("TLabel", background=palette["panel"], foreground=palette["text"])
    style.configure("Hero.TLabel", background=palette["bg"], foreground=palette["gold_hot"], font=("Cambria", 21, "bold"))
    style.configure("Subtitle.TLabel", background=palette["bg"], foreground=palette["subtle"], font=("Segoe UI", 9))
    style.configure("Subtle.TLabel", background=palette["panel_alt"], foreground=palette["subtle"])
    style.configure("Status.TLabel", background=palette["bg"], foreground=palette["subtle"])
    style.configure("Count.TLabel", background=palette["panel_alt"], foreground=palette["gold_hot"], font=("Segoe UI Semibold", 9))

    style.configure("TButton",
                    background=palette["panel_glow"],
                    foreground=palette["text"],
                    bordercolor=palette["border"],
                    focusthickness=0,
                    padding=(12, 7))
    style.map("TButton",
              background=[("active", palette["gold_deep"]), ("pressed", palette["amber"]), ("disabled", "#1B2433")],
              foreground=[("active", palette["gold_hot"]), ("pressed", palette["select_fg"]), ("disabled", "#687184")],
              bordercolor=[("active", palette["gold"]), ("pressed", palette["gold_hot"])])

    style.configure("Accent.TButton",
                    background=palette["gold_deep"],
                    foreground=palette["gold_hot"],
                    bordercolor=palette["gold"],
                    padding=(13, 7))
    style.map("Accent.TButton",
              background=[("active", palette["gold"]), ("pressed", palette["gold_hot"]), ("disabled", "#1B2433")],
              foreground=[("active", palette["select_fg"]), ("pressed", palette["select_fg"]), ("disabled", "#687184")])

    style.configure("TEntry",
                    fieldbackground=palette["entry_bg"],
                    background=palette["entry_bg"],
                    foreground=palette["text"],
                    insertcolor=palette["gold_hot"],
                    bordercolor=palette["border"],
                    lightcolor=palette["border"],
                    darkcolor=palette["border"],
                    padding=5)
    style.map("TEntry",
              bordercolor=[("focus", palette["gold"]), ("active", palette["gold_deep"])])

    style.configure("TCombobox",
                    fieldbackground=palette["entry_bg"],
                    background=palette["panel_glow"],
                    foreground=palette["text"],
                    arrowcolor=palette["gold"],
                    bordercolor=palette["border"],
                    padding=4)
    style.map("TCombobox",
              fieldbackground=[("readonly", palette["entry_bg"])],
              foreground=[("readonly", palette["text"])],
              bordercolor=[("focus", palette["gold"])])

    style.configure("TNotebook", background=palette["bg"], borderwidth=0)
    style.configure("TNotebook.Tab",
                    background=palette["panel_alt"],
                    foreground=palette["subtle"],
                    padding=(16, 7),
                    borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", palette["panel"]), ("active", palette["panel_glow"])],
              foreground=[("selected", palette["gold_hot"]), ("active", palette["gold"])])

    style.configure("Treeview",
                    background=palette["tree_bg"],
                    fieldbackground=palette["tree_bg"],
                    foreground=palette["text"],
                    bordercolor=palette["border"],
                    rowheight=26,
                    font=("Consolas", 10))
    style.configure("Treeview.Heading",
                    background=palette["panel_glow"],
                    foreground=palette["gold_hot"],
                    bordercolor=palette["border"],
                    relief="flat",
                    font=("Segoe UI Semibold", 9))
    style.map("Treeview",
              background=[("selected", palette["select_bg"])],
              foreground=[("selected", palette["select_fg"])])

    style.configure("Vertical.TScrollbar",
                    background=palette["panel_glow"],
                    troughcolor=palette["entry_bg"],
                    bordercolor=palette["border"],
                    arrowcolor=palette["gold"])
    style.configure("Horizontal.TScrollbar",
                    background=palette["panel_glow"],
                    troughcolor=palette["entry_bg"],
                    bordercolor=palette["border"],
                    arrowcolor=palette["gold"])

    style.festum_palette = palette
    return style

class FestumConversionApp(tk.Tk):
    VIRTUAL_SCROLL_LINES = 4

    def __init__(self):
        super().__init__()
        self._style = apply_festum_gold_theme(self)
        self._palette = getattr(self._style, 'festum_palette', {})

        self.title(APP_TITLE)
        self.geometry("1180x740")
        self.minsize(900, 560)

        self.current_file_path = None
        self.current_strings = []
        self.current_meta = None
        self.current_format = None
        self.current_encoding = tk.StringVar(value=DEFAULT_ENCODING)

        self.mod_taildata: bytes = b""
        self.original_core: bytes = b""
        self.work_q = queue.Queue()
        self._busy = False
        self.visible_start = 0
        self.selected_index = None
        self.last_search_needle = ""
        self.last_search_index = -1
        self.render_after_id = None

        self.build_ui()
        self.after(100, self.pollwork_queue)

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        hero = ttk.Frame(self, style="Hero.TFrame")
        hero.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        hero.grid_columnconfigure(1, weight=1)

        sigil = tk.Canvas(hero, width=84, height=62, highlightthickness=0, bd=0, bg=self._palette.get("bg", "#07111F"))
        sigil.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))
        self.draw_festum_sigil(sigil)

        ttk.Label(hero, text="Festum Conversion", style="Hero.TLabel").grid(row=0, column=1, sticky="sw")
        ttk.Label(hero, text="Binary text translator for XL, ECB, StringTable, EM, and MESC files", style="Subtitle.TLabel").grid(row=1, column=1, sticky="nw")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(hero, textvariable=self.status_var, style="Status.TLabel", anchor="e").grid(row=0, column=2, rowspan=2, sticky="e", padx=(18, 0))

        top = ttk.Frame(self, style="Toolbar.TFrame")
        top.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        top.grid_columnconfigure(10, weight=1)

        ttk.Button(top, text="Open File", command=self.open_file, style="Accent.TButton").grid(row=0, column=0, padx=(10, 4), pady=10)

        self.save_btn = ttk.Button(top, text="Save", command=self.save_file, state="disabled")
        self.save_btn.grid(row=0, column=1, padx=4, pady=10)

        ttk.Button(top, text="Save As", command=self.save_file_as).grid(row=0, column=2, padx=4, pady=10)
        ttk.Separator(top, orient="vertical").grid(row=0, column=3, sticky="ns", padx=10, pady=10)
        ttk.Button(top, text="Export JSON", command=self.export_json).grid(row=0, column=4, padx=4, pady=10)
        ttk.Button(top, text="Import JSON", command=self.import_json).grid(row=0, column=5, padx=4, pady=10)
        ttk.Separator(top, orient="vertical").grid(row=0, column=6, sticky="ns", padx=10, pady=10)

        ttk.Label(top, text="Encoding", style="Subtle.TLabel").grid(row=0, column=7, padx=(0, 5), pady=10)
        enc = ttk.Combobox(top, textvariable=self.current_encoding, values=["utf-8", "cp932", "shift_jis", "big5", "koei-en"], width=11, state="readonly")
        enc.grid(row=0, column=8, padx=(0, 10), pady=10)

        self.count_var = tk.StringVar(value="0 strings")
        ttk.Label(top, textvariable=self.count_var, style="Count.TLabel", anchor="e").grid(row=0, column=10, sticky="e", padx=10, pady=10)

        self.nb = ttk.Notebook(self)
        self.nb.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self.tab_strings = ttk.Frame(self.nb)

        self.nb.add(self.tab_strings, text="Strings")

        self.build_strings_tab(self.tab_strings)

    def draw_festum_sigil(self, canvas: tk.Canvas):
        pal = self._palette
        bg = pal.get("bg", "#07111F")
        gold = pal.get("gold", "#E6C66A")
        hot = pal.get("gold_hot", "#FFE8A3")
        deep = pal.get("gold_deep", "#A87925")
        canvas.delete("all")
        canvas.create_oval(18, 5, 66, 53, outline=deep, width=2)
        canvas.create_oval(27, 14, 57, 44, outline=gold, width=2)
        canvas.create_line(42, 0, 42, 58, fill=hot, width=1)
        canvas.create_line(4, 31, 80, 31, fill=gold, width=1)
        canvas.create_line(12, 12, 72, 50, fill=deep, width=1)
        canvas.create_line(72, 12, 12, 50, fill=deep, width=1)
        canvas.create_oval(35, 24, 49, 38, fill=hot, outline=bg)

    def build_strings_tab(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        controls = ttk.Frame(parent, style="Panel.TFrame")
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        controls.grid_columnconfigure(4, weight=1)

        ttk.Label(controls, text="Search", style="TLabel").grid(row=0, column=0, padx=(10, 5), pady=8)
        self.search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(controls, textvariable=self.search_var, width=40)
        search_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=8)
        ttk.Button(controls, text="Find Next", command=self.find_next).grid(row=0, column=2, padx=4, pady=8)
        ttk.Label(controls, text="Double-click a visible row to edit. The table only renders the current viewport.", style="Subtle.TLabel").grid(row=0, column=4, sticky="e", padx=10, pady=8)

        cols = ("Index", "Text")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings")
        self.tree.heading("Index", text="Index")
        self.tree.heading("Text", text="Text")
        self.tree.column("Index", width=100, anchor="e", stretch=False)
        self.tree.column("Text", width=980, anchor="w")
        self.tree.tag_configure("odd", background=self._palette.get("tree_alt", "#101B2D"))
        self.tree.tag_configure("even", background=self._palette.get("tree_bg", "#0B1422"))

        self.vsb = ttk.Scrollbar(parent, orient="vertical", command=self.on_virtual_scroll)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=hsb.set)

        self.tree.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 0))
        self.vsb.grid(row=1, column=1, sticky="ns", padx=(0, 10), pady=(0, 0))
        hsb.grid(row=2, column=0, sticky="ew", padx=(10, 0), pady=(0, 10))

        self.tree.bind("<Double-1>", self.on_edit_cell)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Configure>", self.schedule_virtual_render)
        self.tree.bind("<MouseWheel>", self.on_tree_mousewheel)
        self.tree.bind("<Button-4>", self.on_tree_mousewheel)
        self.tree.bind("<Button-5>", self.on_tree_mousewheel)
        self.tree.bind("<KeyPress>", self.on_tree_key)

    # Status/threading

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.save_btn.configure(state=("disabled" if busy or not self.current_meta else "normal"))

    def run_worker(self, fn, *args):
        if self._busy:
            return
        self.set_busy(True)

        def worker():
            try:
                res = fn(*args)
                self.work_q.put(("ok", res))
            except Exception as e:
                self.work_q.put(("err", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def pollwork_queue(self):
        try:
            kind, payload = self.work_q.get_nowait()
        except queue.Empty:
            self.after(100, self.pollwork_queue)
            return

        self.set_busy(False)

        if kind == "err":
            messagebox.showerror("Error", payload)
            self.set_status("Error.")
        else:
            action, data = payload
            if action == "load":
                strings, meta, fmt, path, data_core, mod_tail = data
                self.on_loaded(path, fmt, strings, meta, data_core, mod_tail)
            elif action == "save":
                out_path = data
                self.set_status(f"Saved: {os.path.basename(out_path)}")
            else:
                self.set_status("Done.")

        self.after(100, self.pollwork_queue)

    # Load/Save

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open supported file",
            filetypes=[
                ("Supported Festum files", SUPPORTED_FILE_PATTERNS),
                ("XL files", "*.xl"),
                ("EM files", "*.em"),
                ("MESC files", "*.mesc"),
                ("ECB files", "*.ecb"),
                ("Possible binary text tables", "*.bin *.dat *.tbl"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        allowed_exts = {".xl", ".em", ".mesc", ".ecb", ".bin", ".dat", ".tbl"}
        ext = os.path.splitext(path)[1].lower()

        if ext not in allowed_exts:
            messagebox.showwarning(
                "Unsupported file",
                f"Unsupported file extension: {ext or '(none)'}\n\n"
                "Supported extensions are:\n"
                ".xl, .em, .mesc, .ecb, .bin, .dat, .tbl"
            )
            return

        encoding = self.current_encoding.get()

        def do_load(pth, enc):
            with open(pth, "rb") as f:
                data_full = f.read()

            data_core, mod_tail = split_mod_taildata(data_full)

            def parse_with(data_bytes: bytes):
                fmt_local = detect_format(pth, data_bytes)
                if not fmt_local:
                    raise ValueError("Unknown/unsupported file format (no known signatures)")

                if fmt_local == "em":
                    strings_local, meta_local = parse_em(data_bytes, enc)
                elif fmt_local == "mesc":
                    strings_local, meta_local = parse_mesc(data_bytes, enc)
                elif fmt_local == "xl19":
                    strings_local, meta_local = parse_xl19(data_bytes, enc)
                elif fmt_local == "xl_legacy":
                    strings_local, meta_local = parse_xl_legacy(data_bytes, enc)
                elif fmt_local == "ecb":
                    strings_local, meta_local = parse_ecb(data_bytes, enc)
                elif fmt_local == "stringtable":
                    strings_local, meta_local = parse_stringtable(data_bytes, enc)
                else:
                    raise ValueError(f"Unsupported detected format: {fmt_local}")

                return fmt_local, strings_local, meta_local

            fmt, strings, meta = parse_with(data_core)

            return ("load", (strings, meta, fmt, pth, data_core, mod_tail))

        self.set_status(f"Loading {os.path.basename(path)}")
        self.run_worker(do_load, path, encoding)

    def on_loaded(self, path, fmt, strings, meta, data_core=b'', mod_tail=b''):
        self.current_file_path = path
        self.current_format = fmt
        self.current_strings = strings
        self.current_meta = meta


        self.original_core = data_core if data_core is not None else b""
        self.mod_taildata = mod_tail if mod_tail is not None else b""
        self.populate_tree(strings)
        self.save_btn.configure(state="normal")
        self.set_status(f"{os.path.basename(path)} – {len(strings)} strings loaded ({fmt})." + (" [taildata]" if self.mod_taildata else ""))

    def populate_tree(self, strings):
        self.visible_start = 0
        self.selected_index = None
        self.last_search_needle = ""
        self.last_search_index = -1
        self.render_virtual_rows()

    def get_virtual_page_size(self) -> int:
        try:
            rowheight = int(self._style.lookup("Treeview", "rowheight") or 26)
        except Exception:
            rowheight = 26
        height = max(1, self.tree.winfo_height())
        return max(1, height // max(1, rowheight))

    def max_virtual_start(self) -> int:
        return max(0, len(self.current_strings) - self.get_virtual_page_size())

    def set_virtual_start(self, start: int):
        self.visible_start = max(0, min(int(start), self.max_virtual_start()))
        self.render_virtual_rows()

    def schedule_virtual_render(self, event=None):
        if self.render_after_id is not None:
            try:
                self.after_cancel(self.render_after_id)
            except Exception:
                pass
        self.render_after_id = self.after(25, self.render_virtual_rows)

    def update_virtual_scrollbar(self):
        total = len(self.current_strings)
        if total <= 0:
            self.vsb.set(0, 1)
            self.count_var.set("0 strings")
            return

        page = self.get_virtual_page_size()
        start = max(0, min(self.visible_start, self.max_virtual_start()))
        end = min(total, start + page)
        first = start / total
        last = min(1.0, end / total)
        self.vsb.set(first, last)
        self.count_var.set(f"{total:,} strings | showing {start + 1:,}-{end:,}")

    def render_virtual_rows(self):
        self.render_after_id = None
        total = len(self.current_strings)
        page = self.get_virtual_page_size()
        self.visible_start = max(0, min(self.visible_start, self.max_virtual_start()))
        start = self.visible_start
        end = min(total, start + page)

        self.tree.delete(*self.tree.get_children(""))
        for idx in range(start, end):
            tag = "even" if idx % 2 == 0 else "odd"
            self.tree.insert("", "end", iid=f"row-{idx}", values=(str(idx), self.current_strings[idx]), tags=(tag,))

        if self.selected_index is not None and start <= self.selected_index < end:
            iid = f"row-{self.selected_index}"
            self.tree.selection_set(iid)
            self.tree.focus(iid)

        self.update_virtual_scrollbar()

    def focus_string_index(self, idx: int):
        total = len(self.current_strings)
        if total <= 0:
            return
        idx = max(0, min(int(idx), total - 1))
        page = self.get_virtual_page_size()
        if not (self.visible_start <= idx < self.visible_start + page):
            self.visible_start = max(0, min(idx - max(0, page // 2), self.max_virtual_start()))
        self.selected_index = idx
        self.render_virtual_rows()
        iid = f"row-{idx}"
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)

    def on_virtual_scroll(self, *args):
        if not self.current_strings:
            return
        page = self.get_virtual_page_size()
        if args[0] == "moveto":
            fraction = float(args[1])
            self.set_virtual_start(round(fraction * len(self.current_strings)))
        elif args[0] == "scroll":
            amount = int(args[1])
            unit = args[2]
            delta = amount if unit == "units" else amount * page
            self.set_virtual_start(self.visible_start + delta)

    def on_tree_mousewheel(self, event):
        if not self.current_strings:
            return "break"
        if getattr(event, "num", None) == 4:
            delta = -self.VIRTUAL_SCROLL_LINES
        elif getattr(event, "num", None) == 5:
            delta = self.VIRTUAL_SCROLL_LINES
        else:
            delta = -int(event.delta / 120) * self.VIRTUAL_SCROLL_LINES
        self.set_virtual_start(self.visible_start + delta)
        return "break"

    def on_tree_key(self, event):
        if not self.current_strings:
            return
        page = self.get_virtual_page_size()
        current = self.selected_index
        if current is None:
            current = self.visible_start

        if event.keysym == "Down":
            self.focus_string_index(current + 1)
        elif event.keysym == "Up":
            self.focus_string_index(current - 1)
        elif event.keysym == "Next":
            self.focus_string_index(current + page)
        elif event.keysym == "Prior":
            self.focus_string_index(current - page)
        elif event.keysym == "Home":
            self.focus_string_index(0)
        elif event.keysym == "End":
            self.focus_string_index(len(self.current_strings) - 1)
        else:
            return
        return "break"

    def on_tree_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        values = self.tree.item(sel[0], "values")
        if values:
            self.selected_index = safe_int(values[0], self.selected_index if self.selected_index is not None else 0)

    def save_file(self):
        if not self.current_file_path or not self.current_meta:
            return
        self.save_to_path(self.current_file_path)

    def save_file_as(self):
        if not self.current_meta:
            messagebox.showinfo("Save As", "Open a file first.")
            return
        base = os.path.basename(self.current_file_path) if self.current_file_path else "output.bin"
        path = filedialog.asksaveasfilename(title="Save As", initialfile=base, defaultextension=os.path.splitext(base)[1] or ".bin")
        if not path:
            return
        self.save_to_path(path)

    def save_to_path(self, out_path: str):
        encoding = self.current_encoding.get()


        def do_save(src_path, dst_path, fmt, strings, meta, enc):
            # Use the bytes we loaded to avoid re-parsing issues,
            # then re-append the mod taildata as the final 6 bytes
            original = self.original_core if self.original_core else b""
            mod_tail = self.mod_taildata if self.mod_taildata else b""
            if not original:
                # Fallback, read from disk, will not preserve mod taildata reliably if the file changed externally
                with open(src_path, "rb") as f:
                    data_full = f.read()
                original, mod_tail = split_mod_taildata(data_full)

            if fmt == "em":
                out = write_em(original, strings, meta, enc)
            elif fmt == "mesc":
                out = write_mesc(original, strings, meta, enc)
            elif fmt == "xl19":
                out = write_xl19(original, strings, meta, enc)
            elif fmt == "xl_legacy":
                out = write_xl_legacy(original, strings, meta, enc)
            elif fmt == "ecb":
                out = write_ecb(original, strings, meta, enc)
            elif fmt == "stringtable":
                out = write_stringtable(original, strings, meta, enc)
            else:
                raise ValueError(f"Unsupported save format: {fmt}")

            # Ensure Aldnoah Engine taildata is the last 6 bytes of the output
            if mod_tail:
                if out.endswith(mod_tail):
                    out = out[:-MOD_TAIL_SIZE]
                out = out + mod_tail

            with open(dst_path, "wb") as fo:
                fo.write(out)
            return ("save", dst_path)

        self.set_status(f"Saving {os.path.basename(out_path)}")
        self.run_worker(do_save, self.current_file_path, out_path, self.current_format, self.current_strings, self.current_meta, encoding)

    # Editing

    def on_edit_cell(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        idx_str, old = self.tree.item(item, "values")
        idx = safe_int(idx_str, -1)
        if idx < 0:
            return

        win = tk.Toplevel(self)
        win.title(f"Edit String #{idx}")
        win.geometry("760x220")
        win.configure(bg=self._palette.get("bg", "#07111F"))
        win.transient(self)
        win.grab_set()

        txt = tk.Text(win, wrap="word", height=8)
        try:
            pal = getattr(self, "_palette", {})
            txt.configure(
                bg=pal.get("entry_bg", "#07101C"),
                fg=pal.get("text", "#F8EBC4"),
                insertbackground=pal.get("gold_hot", "#FFE8A3"),
                selectbackground=pal.get("select_bg", "#D9AD4E"),
                selectforeground=pal.get("select_fg", "#08101C"),
                relief="flat",
                padx=10,
                pady=8
            )
        except Exception:
            pass
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert("1.0", self.current_strings[idx])
        txt.focus_set()

        def on_ok():
            new = txt.get("1.0", "end-1c")
            self.current_strings[idx] = new
            self.selected_index = idx
            self.render_virtual_rows()
            self.set_status(f"Edited string #{idx}.")
            win.destroy()

        btns = ttk.Frame(win, style="Hero.TFrame")
        btns.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(btns, text="OK", command=on_ok, style="Accent.TButton").pack(side="right", padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=4)
        win.bind("<Control-Return>", lambda _e: on_ok())
        win.bind("<Escape>", lambda _e: win.destroy())

    def find_next(self):
        needle = self.search_var.get()
        if not needle:
            return
        total = len(self.current_strings)
        if total <= 0:
            return

        needle_l = needle.lower()
        if needle_l == self.last_search_needle and self.last_search_index >= 0:
            start = (self.last_search_index + 1) % total
        elif self.selected_index is not None:
            start = (self.selected_index + 1) % total
        else:
            start = 0

        for step in range(total):
            idx = (start + step) % total
            if needle_l in str(self.current_strings[idx]).lower():
                self.last_search_needle = needle_l
                self.last_search_index = idx
                self.focus_string_index(idx)
                self.set_status(f"Found '{needle}' at string #{idx}.")
                return
        messagebox.showinfo("Search", "No more matches.")

    # Import/Export

    def export_json(self):
        if not self.current_meta:
            return
        path = filedialog.asksaveasfilename(title="Export strings JSON", defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path:
            return
        obj = {
            "file": os.path.basename(self.current_file_path) if self.current_file_path else "",
            "format": self.current_format,
            "encoding": self.current_encoding.get(),
            "strings": self.current_strings,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
            self.set_status(f"Exported: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def import_json(self):
        if not self.current_meta:
            messagebox.showinfo("Import", "Open a file first.")
            return
        path = filedialog.askopenfilename(title="Import strings JSON", filetypes=[("JSON","*.json"), ("All Files","*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            strings = obj.get("strings")
            if not isinstance(strings, list):
                raise ValueError("JSON missing 'strings' array")
            if len(strings) != len(self.current_strings):
                raise ValueError(f"String count mismatch: file has {len(self.current_strings)}, JSON has {len(strings)}")
            self.current_strings = [str(s) for s in strings]
            self.populate_tree(self.current_strings)
            self.set_status(f"Imported: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

if __name__ == "__main__":
    app = FestumConversionApp()
    app.mainloop()
