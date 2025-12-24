import os, re, struct, threading, queue, time, ctypes, json
import wave as _wave
import array as _array
import math as _math
import sys as _sys
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

DECODE_MSADPCM_TO_PCM_WAV = True  # recommended for maximum WAV player compatibility


"""
Kybernes Scanner, WBH/WBD Tool (Tkinter + Threading)

Tab 1: Scan for wrapped WBH/WBD banks and unpack to <stem>_WB folders
        + OPTIONAL: extract KWB2 subsongs directly from WBH metadata into numbered raw chunks and MS-ADPCM WAVs

Tab 2: Re-wrap from <stem>_WB folders into new .bin files in a central New_WBS folder
        + OPTIONAL: rebuild WBD from extracted subsongs and rewrite WBH stream_offset/stream_size accordingly

Signatures:
  _HBW0000  (WBH)  5F 48 42 57 30 30 30 30
  _DBW0000  (WBD)  5F 44 42 57 30 30 30 30

Wrapper layout:
  0x00: u32 wbh_off
  0x04: u32 wbh_size
  0x08: u32 wbd_off
  0x0C: u32 wbd_size
  0x10: WBH blob
  ... : WBD blob
  ... : taildata (last 6 bytes, used by Mod Manager/Aldnoah Engine)

Taildata:
  Always preserved from the original wrapped file
  This tool does NOT expose taildata editing on purpose
"""

SIG_WBH = bytes.fromhex("5F 48 42 57 30 30 30 30")  # _HBW0000
SIG_WBD = bytes.fromhex("5F 44 42 57 30 30 30 30")  # _DBW0000

TAIL_LEN = 6  # always 6 bytes

# helpers/structs

@dataclass
class ParseResult:
    ok: bool
    pairs_extracted: int = 0
    message: str = ""

@dataclass
class SubsongInfo:
    index: int
    sound_index: int
    subsound_index: int
    wbh_stream_off_pos: int   # absolute position inside WBH blob for stream_offset field (u32le)
    wbh_stream_size_pos: int  # absolute position inside WBH blob for stream_size field (u32le)
    wbh_num_samples_pos: int  # absolute position inside WBH blob for num_samples field (u32le)
    stream_offset: int        # offset into WBD payload (WBD data start is at 0x0C)
    stream_size: int
    sample_rate: int
    codec: int
    channels: int
    block_align: int          # MSADPCM block size
    num_samples: int


    wbd_file_offset: int = 0     # absolute file offset inside WBD blob for this subsong's data start
    wbd_base_offset: int = 0     # base offset added to WBH stream_offset to get file offset
    dsp_header_pos: int = -1   # absolute position inside WBH for DSP header blob (codec 0x90)
    dsp_big_endian: bool = False
    pcm_layout: str = "auto"  # auto/interleaved/planar (PCM16 only)
def unique_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path
    i = 1
    while True:
        candidate = f"{base_path}_{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def natural_sort_key(s: str):
    parts = re.split(r"(\d+)", s)
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return key


def stream_find_signatures(path: str, sigs: list[bytes], chunk_size: int = 1024 * 1024, max_scan_bytes: int | None = None) -> dict[bytes, int | None]:
    found: dict[bytes, int | None] = {s: None for s in sigs}
    remaining = set(sigs)

    overlap = max(len(s) for s in sigs) - 1
    scanned = 0
    tail = b""

    try:
        with open(path, "rb") as f:
            while remaining:
                if max_scan_bytes is not None and scanned >= max_scan_bytes:
                    break

                to_read = chunk_size
                if max_scan_bytes is not None:
                    to_read = min(to_read, max_scan_bytes - scanned)

                chunk = f.read(to_read)
                if not chunk:
                    break

                buf = tail + chunk

                for s in list(remaining):
                    idx = buf.find(s)
                    if idx != -1:
                        abs_off = (scanned - len(tail)) + idx
                        found[s] = abs_off
                        remaining.remove(s)

                tail = buf[-overlap:] if len(buf) > overlap else buf
                scanned += len(chunk)

    except OSError:
        return found

    return found


def _patch_internal_size(blob: bytes, sig: bytes, new_size: int) -> bytes:
    if len(blob) < 12:
        raise ValueError("Blob too small to patch size field.")
    if blob[0:8] != sig:
        raise ValueError("Signature mismatch; cannot patch size field.")
    return blob[:8] + struct.pack("<I", new_size) + blob[12:]



# MS ADPCM WAV helpers
# We generate standard RIFF/WAVE files so modders can preview/edit without running vgmstream thousands of times

MSADPCM_FORMAT_TAG = 0x0002

# Standard MSADPCM coefficient table (7 pairs)
MSADPCM_COEFS = [
    (256,   0),
    (512, -256),
    (0,     0),
    (192,  64),
    (240,   0),
    (460, -208),
    (392, -232),
]

def msadpcm_samples_per_block(block_align: int, channels: int) -> int:
    if channels <= 0:
        raise ValueError("Invalid channels for MSADPCM.")
    base = block_align - (7 * channels)
    if base < 0:
        raise ValueError("Invalid block_align for MSADPCM.")
    return (base * 2 // channels) + 2


# MSADPCM adaptation table (standard)
MSADPCM_ADAPTATION_TABLE = [
    230,230,230,230,307,409,512,614,
    768,614,512,409,307,230,230,230
]

def _msadpcm_nibble_to_signed(n: int) -> int:
    return n if n < 8 else n - 16

def decode_msadpcm_to_pcm16(adpcm: bytes, sample_rate: int, channels: int, block_align: int, num_samples: int) -> bytes:
    """
    Decode Microsoft 4-bit ADPCM blocks to PCM16LE
    This matches what vgmstream outputs and avoids playback issues in players that don't support MSADPCM WAV
    - channels: 1 or 2 supported
    - num_samples: total samples per channel
    Returns PCM16LE interleaved (L,R,L,R,etc) for stereo
    """
    if channels not in (1, 2):
        raise ValueError("MSADPCM decode supports only 1 or 2 channels for now.")
    if block_align <= 0:
        raise ValueError("Invalid block_align for MSADPCM decode.")

    spb = msadpcm_samples_per_block(block_align, channels)
    if num_samples <= 0:
        # derive from whole blocks
        num_samples = (len(adpcm) // block_align) * spb

    # standard coefs, indexed by predictor (0..6)
    coef1 = [a for a, _ in MSADPCM_COEFS]
    coef2 = [b for _, b in MSADPCM_COEFS]

    out = bytearray()
    total_blocks = len(adpcm) // block_align
    samples_written = 0  # per channel

    for bi in range(total_blocks):
        if samples_written >= num_samples:
            break

        base = bi * block_align

        if channels == 1:
            pred = adpcm[base + 0]
            if pred > 6:
                break
            c1, c2 = coef1[pred], coef2[pred]
            delta = struct.unpack_from("<H", adpcm, base + 1)[0]
            if delta == 0:
                delta = 16
            s1 = struct.unpack_from("<h", adpcm, base + 3)[0]
            s2 = struct.unpack_from("<h", adpcm, base + 5)[0]

            # output initial samples (sample2 then sample1)
            if samples_written < num_samples:
                out += struct.pack("<h", s2); samples_written += 1
            if samples_written < num_samples:
                out += struct.pack("<h", s1); samples_written += 1

            prev1, prev2 = s1, s2
            pos = base + 7
            # each byte gives 2 samples
            while pos < base + block_align and samples_written < num_samples:
                b = adpcm[pos]; pos += 1
                for nib in ((b >> 4) & 0x0F, b & 0x0F):  # high then low
                    if samples_written >= num_samples:
                        break
                    sn = _msadpcm_nibble_to_signed(nib)
                    predicted = (c1 * prev1 + c2 * prev2) >> 8
                    sample = predicted + sn * delta
                    if sample > 32767: sample = 32767
                    if sample < -32768: sample = -32768
                    out += struct.pack("<h", sample)
                    prev2, prev1 = prev1, sample
                    delta = (MSADPCM_ADAPTATION_TABLE[nib] * delta) >> 8
                    if delta < 16:
                        delta = 16

        else:
            # stereo
            pred0 = adpcm[base + 0] % 7
            pred1 = adpcm[base + 1] % 7
            c10, c20 = coef1[pred0], coef2[pred0]
            c11, c21 = coef1[pred1], coef2[pred1]

            delta0 = struct.unpack_from("<H", adpcm, base + 2)[0]
            delta1 = struct.unpack_from("<H", adpcm, base + 4)[0]


            if delta0 == 0: delta0 = 16
            if delta1 == 0: delta1 = 16
            s10 = struct.unpack_from("<h", adpcm, base + 6)[0]
            s11 = struct.unpack_from("<h", adpcm, base + 8)[0]
            s20 = struct.unpack_from("<h", adpcm, base + 10)[0]
            s21 = struct.unpack_from("<h", adpcm, base + 12)[0]

            # output initial samples (sample2 then sample1)
            if samples_written < num_samples:
                out += struct.pack("<hh", s20, s21); samples_written += 1
            if samples_written < num_samples:
                out += struct.pack("<hh", s10, s11); samples_written += 1

            prev10, prev20 = s10, s20
            prev11, prev21 = s11, s21

            pos = base + 14
            while pos < base + block_align and samples_written < num_samples:
                b = adpcm[pos]; pos += 1
                nib0 = (b >> 4) & 0x0F
                nib1 = b & 0x0F

                sn0 = _msadpcm_nibble_to_signed(nib0)
                sn1 = _msadpcm_nibble_to_signed(nib1)

                pred0v = (c10 * prev10 + c20 * prev20) >> 8
                samp0 = pred0v + sn0 * delta0
                if samp0 > 32767: samp0 = 32767
                if samp0 < -32768: samp0 = -32768

                pred1v = (c11 * prev11 + c21 * prev21) >> 8
                samp1 = pred1v + sn1 * delta1
                if samp1 > 32767: samp1 = 32767
                if samp1 < -32768: samp1 = -32768

                out += struct.pack("<hh", samp0, samp1)
                samples_written += 1

                prev20, prev10 = prev10, samp0
                prev21, prev11 = prev11, samp1

                delta0 = (MSADPCM_ADAPTATION_TABLE[nib0] * delta0) >> 8
                delta1 = (MSADPCM_ADAPTATION_TABLE[nib1] * delta1) >> 8
                if delta0 < 16: delta0 = 16
                if delta1 < 16: delta1 = 16

    return bytes(out)


def msadpcm_best_skip(raw: bytes, channels: int, block_align: int, max_shift: int = 64, blocks_check: int = 64) -> int:
    """Find a small header skip inside a supposed MSADPCM stream

    Some Koei Tecmo banks include a small per-stream header before ADPCM blocks
    We brute-force a small shift and pick the one that best aligns block headers
    (predictor byte in [0..6] at start of each channel header every block)

    Returns 0 if data already looks aligned or if no confident shift is found
    """
    if not raw or block_align <= 0:
        return 0
    channels = max(1, int(channels))
    block_align = int(block_align)

    max_shift = max(0, min(int(max_shift), len(raw) - 1))
    best_shift = 0
    best_ok = -1
    best_blocks = 0

    for shift in range(max_shift + 1):
        avail = len(raw) - shift
        blocks = min(int(blocks_check), avail // block_align)
        if blocks <= 0:
            continue

        ok = 0
        base = shift
        for bi in range(blocks):
            off = base + bi * block_align
            # predictor is first byte in each channel's 7 byte header
            valid = True
            for ch in range(channels):
                p = raw[off + ch * 7]
                if p >= 7:
                    valid = False
                    break
            if valid:
                ok += 1

        # prefer smaller shift when tied
        if ok > best_ok or (ok == best_ok and shift < best_shift):
            best_ok = ok
            best_shift = shift
            best_blocks = blocks

        # perfect score, can't do better
        if ok == blocks and blocks >= 8 and shift == 0:
            return 0

    # Require confidence: almost all checked blocks must look valid
    if best_blocks >= 8 and best_ok >= int(best_blocks * 0.95):
        return best_shift
    if best_blocks >= 4 and best_ok == best_blocks:
        return best_shift

    return 0
def build_msadpcm_wav(adpcm_data: bytes, sample_rate: int, channels: int, block_align: int, num_samples: int) -> bytes:
    if sample_rate <= 0 or channels <= 0 or block_align <= 0:
        raise ValueError("Invalid WAV parameters.")
    if num_samples < 0:
        raise ValueError("Invalid num_samples.")
    spb = msadpcm_samples_per_block(block_align, channels)
    avg_bps = int(sample_rate * block_align / spb)

    cbSize = 2 + 2 + (len(MSADPCM_COEFS) * 4)  # 32
    fmt_body = struct.pack(
        "<HHIIHHH",
        MSADPCM_FORMAT_TAG,
        channels,
        sample_rate,
        avg_bps,
        block_align,
        4,
        cbSize
    )
    extra = struct.pack("<HH", spb, len(MSADPCM_COEFS))
    for a, b in MSADPCM_COEFS:
        extra += struct.pack("<hh", a, b)

    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_body) + len(extra)) + fmt_body + extra
    fact_chunk = b"fact" + struct.pack("<I", 4) + struct.pack("<I", int(num_samples))
    data_chunk = b"data" + struct.pack("<I", len(adpcm_data)) + adpcm_data

    riff_size = 4 + len(fmt_chunk) + len(fact_chunk) + len(data_chunk)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + fact_chunk + data_chunk
def build_pcm16_wav(pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("Invalid WAV parameters.")
    bits_per_sample = 16
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align

    fmt_body = struct.pack(
        "<HHIIHH",
        0x0001,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
    data_chunk = b"data" + struct.pack("<I", len(pcm_data)) + pcm_data
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk

# PCM16 layout handling (flat vs interleave vs planar)

def _pcm16_smoothness_score_interleaved(pcm: bytes, channels: int, max_samples: int = 4096) -> int:
    """Lower is smoother/more plausible, Operates on channel 0 only for speed"""
    if channels <= 0:
        return 10**18
    step = channels * 2
    n = min((len(pcm) // step), max_samples)
    if n <= 3:
        return 10**18
    # read channel0 samples
    prev = _s16le(pcm, 0)
    score = 0
    off = step
    for _ in range(1, n):
        cur = _s16le(pcm, off)
        score += abs(cur - prev)
        prev = cur
        off += step
    return score

def pcm16_planar_to_interleaved(pcm: bytes, channels: int) -> bytes:
    """Convert planar PCM16 (all samples of ch0 then ch1...) into standard interleaved PCM16"""
    if channels <= 1:
        return pcm
    if len(pcm) % 2 != 0:
        return pcm
    total_samples = len(pcm) // 2
    if total_samples % channels != 0:
        return pcm
    samples_per_ch = total_samples // channels
    # Split planes as lists of int16
    planes = []
    off = 0
    for _ in range(channels):
        plane = struct.unpack_from("<" + "h"*samples_per_ch, pcm, off)
        planes.append(plane)
        off += samples_per_ch * 2
    # Interleave
    out = bytearray(len(pcm))
    o = 0
    for i in range(samples_per_ch):
        for ch in range(channels):
            struct.pack_into("<h", out, o, planes[ch][i])
            o += 2
    return bytes(out)

def normalize_pcm16_for_wav(pcm: bytes, channels: int, mode: str = "auto") -> tuple[bytes, str]:
    """
    Ensure PCM16 bytes are interleaved as WAV expects
    mode:
      interleaved: trust input
      planar: convert from planar to interleaved
      auto: choose by smoothness heuristic (best for 2ch,  falls back to interleaved)
    """
    mode = (mode or "auto").lower()
    if channels <= 1:
        return pcm, "flat"
    if mode == "interleaved":
        return pcm, "interleaved"
    if mode == "planar":
        return pcm16_planar_to_interleaved(pcm, channels), "planar"
    # auto
    if channels == 2 and len(pcm) >= 4096:
        inter_score = _pcm16_smoothness_score_interleaved(pcm, channels)
        planar_pcm = pcm16_planar_to_interleaved(pcm, channels)
        planar_score = _pcm16_smoothness_score_interleaved(planar_pcm, channels)
        if planar_score + 1000 < inter_score:
            return planar_pcm, "planar(auto)"
        return pcm, "interleaved(auto)"
    return pcm, "interleaved(auto)"

# NGC DSP decode support (codec 0x90/DSP_HEAD)

def _s16be(buf: bytes, off: int) -> int:
    return struct.unpack_from(">h", buf, off)[0]

def _s16le(buf: bytes, off: int) -> int:
    return struct.unpack_from("<h", buf, off)[0]

def read_dsp_header_from_wbh(wbh_blob: bytes, dsp_header_pos: int, channels: int, big_endian: bool) -> tuple[list[list[int]], list[tuple[int,int]]]:
    """
    coefs at dsp_offset + 0x1c (16 s16), stride 0x60
    hist  at dsp_offset + 0x40 (2 s16),  stride 0x60
    """
    all_coefs = []
    all_hist = []
    stride = 0x60
    for ch in range(channels):
        base = dsp_header_pos + ch * stride
        if base + 0x44 > len(wbh_blob):
            raise ValueError("DSP header out of range in WBH")
        read_s16 = _s16be if big_endian else _s16le
        coefs = [read_s16(wbh_blob, base + 0x1C + 2*k) for k in range(16)]
        h1 = read_s16(wbh_blob, base + 0x40)
        h2 = read_s16(wbh_blob, base + 0x42)
        all_coefs.append(coefs)
        all_hist.append((h1, h2))
    return all_coefs, all_hist

def decode_dsp_adpcm_mono(data: bytes, num_samples: int, coefs: list[int], hist1: int, hist2: int) -> bytes:
    """
    Decode mono DSP ADPCM frames (8 bytes -> 14 samples), output PCM16LE
    """
    out = bytearray()
    sample_count = 0
    pos = 0
    while pos + 8 <= len(data) and sample_count < num_samples:
        header = data[pos]
        predictor = (header >> 4) & 0x0F
        scale = header & 0x0F
        pos += 1
        if predictor > 7:
            break
        coef1 = coefs[predictor*2 + 0]
        coef2 = coefs[predictor*2 + 1]
        frame = data[pos:pos+7]
        pos += 7
        for b in frame:
            for nib in ((b >> 4) & 0x0F, b & 0x0F):
                if sample_count >= num_samples:
                    break
                s = nib if nib < 8 else nib - 16
                sample = ((s << scale) << 11) + (coef1 * hist1) + (coef2 * hist2)
                sample = sample >> 11
                if sample > 32767: sample = 32767
                if sample < -32768: sample = -32768
                out += struct.pack("<h", sample)
                hist2, hist1 = hist1, sample
                sample_count += 1
    return bytes(out)


def _read_riff_chunks(wav_bytes: bytes) -> dict[str, bytes]:
    if len(wav_bytes) < 12 or wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file.")
    chunks = {}
    off = 12
    while off + 8 <= len(wav_bytes):
        cid = wav_bytes[off:off+4].decode("ascii", errors="replace")
        size = struct.unpack_from("<I", wav_bytes, off+4)[0]
        data_off = off + 8
        data_end = data_off + size
        if data_end > len(wav_bytes):
            break
        chunks[cid] = wav_bytes[data_off:data_end]
        off = data_end + (size & 1)
    return chunks

def parse_msadpcm_wav(wav_path: str) -> dict:
    wav_bytes = open(wav_path, "rb").read()
    chunks = _read_riff_chunks(wav_bytes)
    if "fmt " not in chunks or "data" not in chunks:
        raise ValueError("WAV missing fmt or data chunk.")

    fmt = chunks["fmt "]
    if len(fmt) < 18:
        raise ValueError("fmt chunk too small.")

    wFormatTag, nChannels, nSamplesPerSec, nAvgBytesPerSec, nBlockAlign, wBitsPerSample, cbSize = struct.unpack_from("<HHIIHHH", fmt, 0)
    if wFormatTag != MSADPCM_FORMAT_TAG:
        raise ValueError(f"Unsupported WAV codec: 0x{wFormatTag:04X} (need MS ADPCM).")

    num_samples = None
    if "fact" in chunks and len(chunks["fact"]) >= 4:
        num_samples = struct.unpack_from("<I", chunks["fact"], 0)[0]

    return {
        "sample_rate": int(nSamplesPerSec),
        "channels": int(nChannels),
        "block_align": int(nBlockAlign),
        "num_samples": int(num_samples) if num_samples is not None else None,
        "adpcm_data": chunks["data"],
    }


# WAV (PCM) reader, resample/mix, encoders for repack

def read_pcm_wav_int16(wav_path: str) -> dict:
    """Read a PCM WAV (uncompressed) and return int16 samples as list per channel"""
    with _wave.open(wav_path, "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        ct = wf.getcomptype()
        if ct != "NONE":
            raise ValueError(f"WAV must be uncompressed PCM (got comptype={ct}).")
        if sw != 2:
            raise ValueError(f"WAV must be 16-bit PCM for repack (sampwidth={sw}). Export 16-bit PCM.")
        n = wf.getnframes()
        frames = wf.readframes(n)
    a = _array.array("h")
    a.frombytes(frames)
    if _sys.byteorder != "little":
        a.byteswap()
    if ch <= 0:
        raise ValueError("Invalid channel count.")
    if len(a) != n * ch:
        raise ValueError("Unexpected PCM frame length.")
    per_ch = [a[c::ch].tolist() for c in range(ch)]
    return {"sample_rate": sr, "channels": ch, "num_samples": n, "samples": per_ch}

def _mix_channels(samples: list[list[int]], target_channels: int) -> list[list[int]]:
    src_ch = len(samples)
    if src_ch == target_channels:
        return samples
    n = len(samples[0]) if samples else 0
    if src_ch == 1 and target_channels == 2:
        return [samples[0][:], samples[0][:]]
    if src_ch == 2 and target_channels == 1:
        out = []
        a,b = samples[0], samples[1]
        for i in range(n):
            out.append(int((a[i] + b[i]) / 2))
        return [out]
    # generic: truncate or duplicate last
    if target_channels < src_ch:
        return samples[:target_channels]
    out = [s[:] for s in samples]
    while len(out) < target_channels:
        out.append(out[-1][:])
    return out

def _resample_linear(samples: list[list[int]], src_rate: int, dst_rate: int) -> list[list[int]]:
    """Linear resample for int16 samples per channel"""
    if src_rate == dst_rate:
        return samples
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Invalid sample rate.")
    src_n = len(samples[0]) if samples else 0
    if src_n == 0:
        return samples
    ratio = dst_rate / src_rate
    dst_n = int(_math.floor((src_n) * ratio + 0.5))
    if dst_n <= 0:
        dst_n = 1
    out = []
    for ch_s in samples:
        ch_out = [0] * dst_n
        for i in range(dst_n):
            x = i / ratio
            x0 = int(_math.floor(x))
            x1 = min(x0 + 1, src_n - 1)
            t = x - x0
            s0 = ch_s[x0]
            s1 = ch_s[x1]
            ch_out[i] = int(_math.floor(s0 * (1.0 - t) + s1 * t + 0.5))
        out.append(ch_out)
    return out

def _samples_to_pcm_bytes(samples: list[list[int]]) -> bytes:
    """Interleave per-sample and return little endian PCM16 bytes"""
    ch = len(samples)
    n = len(samples[0]) if ch else 0
    a = _array.array("h")
    # interleave
    for i in range(n):
        for c in range(ch):
            v = samples[c][i]
            if v > 32767: v = 32767
            if v < -32768: v = -32768
            a.append(int(v))
    if _sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()

# MS ADPCM encoder (format 0x0002)

_MSADPCM_COEFS = [(256, 0), (512, -256), (0, 0), (192, 64), (240, 0), (460, -208), (392, -232)]
_MSADPCM_ADAPT = [230,230,230,230,307,409,512,614,768,614,512,409,307,230,230,230]

def _msadpcm_predict(s1: int, s2: int, c1: int, c2: int) -> int:
    return (s1 * c1 + s2 * c2) >> 8

def _msadpcm_quantize(diff: int, delta: int) -> int:
    if delta <= 0:
        return 0
    if diff >= 0:
        q = (diff + (delta // 2)) // delta
    else:
        q = -(((-diff) + (delta // 2)) // delta)
    if q > 7: q = 7
    if q < -8: q = -8
    return int(q)

def encode_msadpcm_block_mono(pcm: list[int], start: int, samples_per_block: int, block_align: int) -> tuple[bytes, int]:
    """Encode one mono MSADPCM block, returns (block_bytes, used_samples_without_padding)"""
    # gather block samples, pad with last value
    rem = len(pcm) - start
    use = min(samples_per_block, rem)
    block = pcm[start:start+use]
    if use < samples_per_block:
        padv = block[-1] if block else 0
        block += [padv] * (samples_per_block - use)

    if samples_per_block < 2:
        raise ValueError("samples_per_block too small")
    s2 = int(block[0])
    s1 = int(block[1])
    best_pred = 0
    best_err = None
    best_codes = None
    best_delta0 = 16
    best_last = None

    # brute force predictor selection (delta0 fixed heuristic)
    for pred in range(7):
        c1, c2 = _MSADPCM_COEFS[pred]
        delta0 = max(16, abs(s1 - s2))
        d = delta0
        ss1, ss2 = s1, s2
        err = 0
        codes = []
        # simulate first up to 32 samples for scoring
        sim_n = min(samples_per_block, 32)
        for i in range(2, sim_n):
            pred_s = _msadpcm_predict(ss1, ss2, c1, c2)
            q = _msadpcm_quantize(int(block[i]) - pred_s, d)
            recon = pred_s + q * d
            if recon > 32767: recon = 32767
            if recon < -32768: recon = -32768
            diff = int(block[i]) - recon
            err += diff * diff
            # update
            d = (d * _MSADPCM_ADAPT[q & 0x0F]) >> 8
            if d < 16: d = 16
            ss2, ss1 = ss1, recon
            codes.append(q)
        if best_err is None or err < best_err:
            best_err = err
            best_pred = pred
            best_delta0 = delta0
            best_last = (ss1, ss2, d)
            best_codes = codes

    # encode full block using chosen predictor
    pred = best_pred
    c1, c2 = _MSADPCM_COEFS[pred]
    delta0 = best_delta0
    d = delta0
    ss2 = int(block[0])
    ss1 = int(block[1])
    codes = []
    for i in range(2, samples_per_block):
        pred_s = _msadpcm_predict(ss1, ss2, c1, c2)
        q = _msadpcm_quantize(int(block[i]) - pred_s, d)
        recon = pred_s + q * d
        if recon > 32767: recon = 32767
        if recon < -32768: recon = -32768
        # update
        d = (d * _MSADPCM_ADAPT[q & 0x0F]) >> 8
        if d < 16: d = 16
        ss2, ss1 = ss1, recon
        codes.append(q)

    # pack header + nibbles
    out = bytearray()
    out.append(pred & 0xFF)
    out += struct.pack("<H", delta0 & 0xFFFF)
    out += struct.pack("<h", int(block[1]))
    out += struct.pack("<h", int(block[0]))

    # encoded bytes count
    enc_bytes = block_align - 7
    # pack 2 nibbles per byte, hi then lo
    nibs = [(c & 0x0F) for c in codes]
    while len(nibs) < enc_bytes * 2:
        nibs.append(0)
    for i in range(enc_bytes):
        hi = nibs[i*2]
        lo = nibs[i*2+1]
        out.append(((hi & 0x0F) << 4) | (lo & 0x0F))
    return bytes(out), use

def encode_msadpcm_block_stereo(pcmL: list[int], pcmR: list[int], start: int, samples_per_block: int, block_align: int) -> tuple[bytes, int]:
    rem = len(pcmL) - start
    use = min(samples_per_block, rem)
    L = pcmL[start:start+use]
    R = pcmR[start:start+use]
    if use < samples_per_block:
        padL = L[-1] if L else 0
        padR = R[-1] if R else 0
        L += [padL]*(samples_per_block-use)
        R += [padR]*(samples_per_block-use)

    # choose predictor per channel
    def choose(pcm):
        s2,s1 = int(pcm[0]), int(pcm[1])
        best=(0, max(16,abs(s1-s2)), None, None)
        best_err=None
        for pred in range(7):
            c1,c2=_MSADPCM_COEFS[pred]
            delta0=max(16,abs(s1-s2))
            d=delta0; ss1=s1; ss2=s2; err=0
            sim_n=min(samples_per_block,32)
            for i in range(2,sim_n):
                pred_s=_msadpcm_predict(ss1,ss2,c1,c2)
                q=_msadpcm_quantize(int(pcm[i])-pred_s,d)
                recon=pred_s+q*d
                if recon>32767: recon=32767
                if recon<-32768: recon=-32768
                diff=int(pcm[i])-recon
                err += diff*diff
                d=(d*_MSADPCM_ADAPT[q&0x0F])>>8
                if d<16: d=16
                ss2,ss1=ss1,recon
            if best_err is None or err<best_err:
                best_err=err
                best=(pred,delta0,s1,s2)
        return best[0], best[1]
    predL, deltaL = choose(L)
    predR, deltaR = choose(R)

    # encode both channels
    def encode_channel(pcm, pred, delta0):
        c1,c2=_MSADPCM_COEFS[pred]
        d=delta0
        ss2=int(pcm[0]); ss1=int(pcm[1])
        codes=[]
        for i in range(2,samples_per_block):
            pred_s=_msadpcm_predict(ss1,ss2,c1,c2)
            q=_msadpcm_quantize(int(pcm[i])-pred_s,d)
            recon=pred_s+q*d
            if recon>32767: recon=32767
            if recon<-32768: recon=-32768
            d=(d*_MSADPCM_ADAPT[q&0x0F])>>8
            if d<16: d=16
            ss2,ss1=ss1,recon
            codes.append(q)
        return codes

    codesL=encode_channel(L,predL,deltaL)
    codesR=encode_channel(R,predR,deltaR)

    out=bytearray()
    # headers: ch0 then ch1
    out.append(predL & 0xFF)
    out += struct.pack("<H", deltaL & 0xFFFF)
    out += struct.pack("<h", int(L[1]))
    out += struct.pack("<h", int(L[0]))
    out.append(predR & 0xFF)
    out += struct.pack("<H", deltaR & 0xFFFF)
    out += struct.pack("<h", int(R[1]))
    out += struct.pack("<h", int(R[0]))

    enc_bytes = block_align - 14
    nibL=[c & 0x0F for c in codesL]
    nibR=[c & 0x0F for c in codesR]
    while len(nibL) < enc_bytes:
        nibL.append(0)
    while len(nibR) < enc_bytes:
        nibR.append(0)
    for i in range(enc_bytes):
        out.append(((nibL[i] & 0x0F) << 4) | (nibR[i] & 0x0F))
    return bytes(out), use

def encode_msadpcm(pcm_samples: list[list[int]], sample_rate: int, channels: int, block_align: int) -> tuple[bytes, int]:
    """Encode PCM16 samples (per channel) into raw MSADPCM blocks"""
    if channels not in (1,2):
        raise ValueError("MSADPCM encoder supports 1 or 2 channels.")
    samples_per_block = msadpcm_samples_per_block(block_align, channels)
    if samples_per_block < 2:
        raise ValueError("Invalid samples_per_block.")
    n = len(pcm_samples[0]) if pcm_samples else 0
    out = bytearray()
    pos = 0
    while pos < n:
        if channels == 1:
            blk, used = encode_msadpcm_block_mono(pcm_samples[0], pos, samples_per_block, block_align)
        else:
            blk, used = encode_msadpcm_block_stereo(pcm_samples[0], pcm_samples[1], pos, samples_per_block, block_align)
        out += blk
        pos += used
    return bytes(out), n

# DSP encoder (NGC DSP ADPCM) for DSP_HEAD banks (mono only by default)

def _encode_dsp_frame(samples14: list[int], coefs: list[int], hist1: int, hist2: int) -> tuple[bytes, int, int]:
    # brute force best predictor/scale
    best_err = None
    best = None

    for pred in range(8):
        c1 = coefs[pred*2+0]
        c2 = coefs[pred*2+1]
        for scale in range(0, 16):
            step = 1 << scale
            h1 = hist1
            h2 = hist2
            err = 0
            nibbles = []
            for s in samples14:
                predicted = ((c1 * h1) + (c2 * h2)) >> 11
                diff = int(s) - predicted
                q = _msadpcm_quantize(diff, step)  # reuse quantizer (same -8..7 rounding)
                recon = predicted + q * step
                if recon > 32767: recon = 32767
                if recon < -32768: recon = -32768
                d = int(s) - recon
                err += d*d
                h2, h1 = h1, recon
                nibbles.append(q & 0x0F)
            if best_err is None or err < best_err:
                best_err = err
                best = (pred, scale, nibbles, h1, h2)

    pred, scale, nibbles, out_h1, out_h2 = best
    header = ((pred & 0x0F) << 4) | (scale & 0x0F)
    # pack 14 nibbles into 7 bytes hi then lo
    b = bytearray()
    b.append(header)
    for i in range(7):
        hi = nibbles[i*2]
        lo = nibbles[i*2+1]
        b.append(((hi & 0x0F) << 4) | (lo & 0x0F))
    return bytes(b), out_h1, out_h2

def encode_dsp_adpcm_mono(pcm: list[int], coefs: list[int], num_samples: int) -> bytes:
    """Encode PCM16 mono samples into DSP ADPCM frames (8 bytes per 14 samples)"""
    out = bytearray()
    h1 = 0
    h2 = 0
    pos = 0
    while pos < num_samples:
        frame = pcm[pos:pos+14]
        if len(frame) < 14:
            frame = frame + [frame[-1] if frame else 0] * (14 - len(frame))
        fb, h1, h2 = _encode_dsp_frame(frame, coefs, h1, h2)
        out += fb
        pos += 14
    return bytes(out)


# KWB2 subsong extraction/rewrite

def _u16le(b: bytes, o: int) -> int:
    return struct.unpack_from("<H", b, o)[0]

def _u32le(b: bytes, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]

def _unique_ints(seq: list[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for v in seq:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _wbd_body_candidates(wbd_blob: bytes) -> list[int]:
    """Generate plausible WBD 'body' start offsets

    Koei Tecmo wavebanks often start with b'_DBW0000' and store some offset at 0x0C
    Unfortunately, different variants appear to place the first audio byte at:
      body_off + 0x00
      body_off + 0x04
      body_off + 0x08

    We return a short candidate set and let scoring pick the best one
    """
    cands: list[int] = [0]

    if len(wbd_blob) >= 0x10 and wbd_blob[:8] == b"_DBW0000":
        body_off = _u32le(wbd_blob, 0x0C)
        for add in (0, 4, 8, 0x10):
            cands.append(body_off + add)

        # some banks behave like: 0x400 + (0/4/8)
        for base in (0x400, 0x800, 0x1000):
            for add in (0, 4, 8):
                cands.append(base + add)

    # clamp to file
    cands = [c for c in _unique_ints(cands) if 0 <= c < len(wbd_blob)]
    return cands

def _score_body_candidate(wbd_blob: bytes, cand: int, raw_meta: list[dict]) -> int:
    """Score a body offset candidate using quick structural checks"""
    score = 0
    wbd_len = len(wbd_blob)

    # check up to N streams (enough to disambiguate, fast)
    for m in raw_meta[:min(24, len(raw_meta))]:
        codec = m["codec"]
        channels = m["channels"]
        block_size = m["block_size"]
        stream_offset = m["stream_offset"]
        stream_size = m["stream_size"]

        file_off = cand + stream_offset
        if file_off < 0 or file_off >= wbd_len:
            score -= 2
            continue

        if file_off + stream_size <= wbd_len:
            score += 1
        else:
            score -= 2
            continue

        if codec == 0x10:  # MSADPCM
            # Prefer candidates that align many consecutive block predictors with minimal skip
            # Some banks include a small per-stream header, so we allow a small shift
            if block_size <= 0:
                score -= 1
                continue

            raw = wbd_blob[file_off:file_off + min(stream_size, block_size * 24 + 64)]
            if len(raw) < block_size * 4:
                score -= 1
                continue

            # Try shifts up to 64 bytes and count valid predictor bytes at each block start
            best_ok = -1
            best_shift = 0
            blocks = min(24, len(raw) // block_size)

            for shift in range(0, min(64, len(raw) - 1) + 1):
                avail = len(raw) - shift
                b = min(blocks, avail // block_size)
                if b <= 0:
                    continue
                ok = 0
                for bi in range(b):
                    off = shift + bi * block_size
                    valid = True
                    for ch in range(max(1, channels)):
                        p = raw[off + ch * 7]
                        if p >= 7:
                            valid = False
                            break
                    if valid:
                        ok += 1
                if ok > best_ok or (ok == best_ok and shift < best_shift):
                    best_ok = ok
                    best_shift = shift
                if ok == b and shift == 0:
                    break

            # reward alignment, penalize big shifts
            score += best_ok * 2
            score -= best_shift // 2

            if stream_size % block_size == 0:
                score += 2
            else:
                score -= 1

        elif codec == 0x00:  # PCM16
            # Very light check: PCM block_align should be channels*2
            # We don't validate content beyond bounds
            if channels in (1, 2, 4, 6, 8):
                score += 1

        else:
            # unknown codec: don't penalize, bounds check already applied
            score += 0

    return score

def detect_wbd_body_offset(wbd_blob: bytes, raw_meta: list[dict] | None = None) -> int:
    """Return file offset where WBD body (audio data) begins

    If raw_meta is provided (a list of parsed WBH subsound entries with codec/offset/size),
    we pick the best candidate by scoring structural validity
    Additionally, if the best candidate consistently requires a fixed small per-stream shift (ex: +0x0C) to align
    MSADPCM blocks, we fold that shift into the returned body offset
    This avoids producing WAVs that start a few bytes before the true ADPCM frame start
    """
    wbd_len = len(wbd_blob)
    if wbd_len < 0x20:
        return 0

    def _guess_candidates() -> list[int]:
        cands: list[int] = [0]
        if wbd_blob[:8] == b"_DBW0000" and wbd_len >= 0x10:
            # Some WBDs store audio immediately after the wrapper header
            cands += [0x0C, 0x10]
            body_off = _u32le(wbd_blob, 0x0C)
            # Koei variants often have real data start at body_off + 0x08 but some banks differ
            for add in (0, 4, 8, 0x0C, 0x10, 0x14, 0x18, 0x20):
                cands.append(body_off + add)
        else:
            # rawish banks: common small headers
            cands += [0, 4, 8, 0x0C, 0x10, 0x14, 0x18]

        # sanitize
        cands = [c for c in cands if 0 <= c < wbd_len]
        return _unique_ints(cands)

    candidates = _guess_candidates()

    if not raw_meta:
        # old behavior: pick something reasonable without WBH guidance
        # Prefer the largest plausible candidate under 0x200 that isn't 0 (to skip tiny headers),
        # but keep 0 as a fallback
        small = [c for c in candidates if c < 0x200]
        if small:
            return max(small)
        return 0

    best = 0
    best_score = -10**9
    for cand in candidates:
        s = _score_body_candidate(wbd_blob, cand, raw_meta)
        if s > best_score:
            best_score = s
            best = cand

    # Normalize consistent MSADPCM alignment shift into the body offset
    # Sometimes our best base is early by a fixed amount (commonly 0x0C),
    # which shows up as best_shift in the MSADPCM predictor checks
    shifts_to_try = (0, 4, 8, 12, 16)
    shift_votes: dict[int, int] = {}

    ms_entries = [m for m in raw_meta[:min(32, len(raw_meta))] if m.get("codec") == 0x10]
    if len(ms_entries) >= 4:
        for m in ms_entries:
            channels = int(m.get("channels", 1) or 1)
            block_size = int(m.get("block_size", 0) or 0)
            stream_offset = int(m.get("stream_offset", 0) or 0)
            stream_size = int(m.get("stream_size", 0) or 0)

            if block_size <= 0:
                continue

            file_off = best + stream_offset
            if file_off < 0 or file_off + min(stream_size, 0x200) > wbd_len:
                continue

            raw = wbd_blob[file_off:file_off + min(stream_size, 0x200)]
            b = min(6, len(raw) // block_size)
            if b <= 0:
                continue

            best_ok = -1
            best_shift = 0
            for shift in shifts_to_try:
                if shift + b * block_size > len(raw):
                    continue
                ok = 0
                for bi in range(b):
                    off = shift + bi * block_size
                    valid = True
                    # predictor byte for each channel at start of channel header
                    for ch in range(max(1, channels)):
                        p = raw[off + ch * 7]
                        if p >= 7:
                            valid = False
                            break
                    if valid:
                        ok += 1
                if ok > best_ok or (ok == best_ok and shift < best_shift):
                    best_ok = ok
                    best_shift = shift
                if ok == b and shift == 0:
                    break

            shift_votes[best_shift] = shift_votes.get(best_shift, 0) + 1

        # pick modal shift
        modal_shift = max(shift_votes.items(), key=lambda kv: kv[1])[0] if shift_votes else 0
        modal_count = shift_votes.get(modal_shift, 0)

        # fold shift into body if it's strongly consistent and non-zero
        if modal_shift != 0 and modal_count >= max(4, int(0.60 * len(ms_entries))):
            adjusted = best + modal_shift
            if 0 <= adjusted < wbd_len:
                adj_score = _score_body_candidate(wbd_blob, adjusted, raw_meta)
                # prefer adjusted if it doesn't get worse (usually improves by removing shift penalty)
                if adj_score >= best_score:
                    return adjusted

    return best

def _p32le(v: int) -> bytes:
    return struct.pack("<I", v)


def parse_kwb2_subsongs(wbh_blob: bytes, wbd_blob: bytes) -> list[SubsongInfo]:
    """Parse WBH (KWB2) + WBD and return subsongs in extraction/rebuild order

    Important: WBD body offset varies across Koei variants, so we:
      parse WBH subsound entries first (offsets/sizes/codecs)
      auto-detect the best WBD body offset using structural scoring
      validate/clamp final file ranges
    """
    subsongs: list[SubsongInfo] = []

    # Locate KWB2 chunk inside WBH wrapper
    kwb_base = wbh_blob.find(b"KWB2")
    if kwb_base < 0:
        raise ValueError("WBH missing KWB2 signature.")

    if kwb_base + 0x18 > len(wbh_blob):
        raise ValueError("WBH too small for KWB2 header.")

    sounds = _u16le(wbh_blob, kwb_base + 0x06)

    # First pass: collect raw meta without needing WBD body offset
    raw_meta: list[dict] = []
    for si in range(sounds):
        so_pos = kwb_base + 0x18 + si * 0x04
        if so_pos + 4 > len(wbh_blob):
            break
        sound_off = _u32le(wbh_blob, so_pos)
        if sound_off == 0:
            continue
        sound_abs = kwb_base + sound_off
        if sound_abs + 0x04 > len(wbh_blob):
            continue

        version = _u16le(wbh_blob, sound_abs + 0x00)
        subsound_count = wbh_blob[sound_abs + 0x03]

        if version < 0xC000:
            subsound_start = 0x2C
            subsound_size = 0x48
        else:
            if sound_abs + 0x30 > len(wbh_blob):
                continue
            subsound_start = _u16le(wbh_blob, sound_abs + 0x2C)
            subsound_size = _u16le(wbh_blob, sound_abs + 0x2E)

        subsound_base = sound_abs + subsound_start
        for ssi in range(subsound_count):
            subsound_abs = subsound_base + ssi * subsound_size
            if subsound_abs + 0x18 > len(wbh_blob):
                continue

            sample_rate = _u16le(wbh_blob, subsound_abs + 0x00)
            codec = wbh_blob[subsound_abs + 0x02]
            channels = wbh_blob[subsound_abs + 0x03]
            block_size = _u16le(wbh_blob, subsound_abs + 0x04)
            num_samples = _u32le(wbh_blob, subsound_abs + 0x0C)
            stream_offset = _u32le(wbh_blob, subsound_abs + 0x10)
            stream_size = _u32le(wbh_blob, subsound_abs + 0x14)
            dsp_header_pos = (subsound_abs + 0x4C) if codec == 0x90 else -1
            dsp_big_endian = False  # Koei PC WBH/WBD are typically little endian, set True if needed

            if stream_size == 0:
                continue

            raw_meta.append({
                "index": len(raw_meta),
                "sound_index": si,
                "subsound_index": ssi,
                "wbh_num_samples_pos": subsound_abs + 0x0C,
                "wbh_stream_off_pos": subsound_abs + 0x10,
                "wbh_stream_size_pos": subsound_abs + 0x14,
                "sample_rate": int(sample_rate),
                "codec": int(codec),
                "channels": int(channels),
                "block_size": int(block_size),
                "num_samples": int(num_samples),
                "stream_offset": int(stream_offset),
                "stream_size": int(stream_size),
                "dsp_header_pos": int(dsp_header_pos),
                "dsp_big_endian": bool(dsp_big_endian),
            })

    # Pick WBD body offset using the raw meta
    wbd_body = detect_wbd_body_offset(wbd_blob, raw_meta)
    wbd_len = len(wbd_blob)

    # Second pass: build SubsongInfo with validated offsets/sizes
    for m in raw_meta:
        codec = m["codec"]
        channels = m["channels"]
        block_size = m["block_size"]
        stream_offset = m["stream_offset"]
        stream_size = m["stream_size"]

        file_off = wbd_body + stream_offset
        if file_off < 0 or file_off >= wbd_len:
            continue
        if file_off + stream_size > wbd_len:
            stream_size = max(0, wbd_len - file_off)
            if stream_size == 0:
                continue

        if codec == 0x10:      # MSADPCM
            block_align = block_size
        elif codec == 0x00:    # PCM16
            block_align = channels * 2
        else:
            # fallback
            block_align = block_size if block_size > 0 else max(1, channels) * 2

        subsongs.append(SubsongInfo(
            index=m["index"],
            sound_index=m["sound_index"],
            subsound_index=m["subsound_index"],
            wbh_stream_off_pos=m["wbh_stream_off_pos"],
            wbh_stream_size_pos=m["wbh_stream_size_pos"],
            wbh_num_samples_pos=m["wbh_num_samples_pos"],
            stream_offset=stream_offset,
            stream_size=stream_size,
            sample_rate=m["sample_rate"],
            codec=codec,
            channels=channels,
            block_align=block_align,
            num_samples=m["num_samples"],
            wbd_file_offset=file_off,
            wbd_base_offset=wbd_body,
        ))
    return subsongs

def extract_subsongs_to_folder(wbh_blob: bytes, wbd_blob: bytes, out_folder: str, prefix: str) -> tuple[int, str]:
    """
    Extract subsongs using WBH metadata
    Writes:
      <out_folder>/<prefix>_subsongs/
        subsongs.json
        0000.bin, 0001.bin, etc
    """
    subsongs = parse_kwb2_subsongs(wbh_blob, wbd_blob)
    if not subsongs:
        return 0, "No KWB2 subsongs found (or unsupported bank)."

    sub_dir = os.path.join(out_folder, f"{prefix}_subsongs")
    os.makedirs(sub_dir, exist_ok=True)
    # WBD base offset is already detected in parse_kwb2_subsongs and stored per subsong
    wbd_body_offset = 0
    wbd_body = wbd_blob

    manifest = []
    for s in subsongs:
        chunk = wbd_blob[s.wbd_file_offset:s.wbd_file_offset + s.stream_size]
        fn = f"{s.index:04d}.bin"
        wav_fn = f"{s.index:04d}.wav"
        with open(os.path.join(sub_dir, fn), "wb") as f:
            f.write(chunk)

        try:
            if s.codec == 0x10:
                # Some banks include a small per-stream header before MSADPCM blocks
                # We keep .bin exactly as is for repacking but skip/trim for WAV decoding if needed
                wav_chunk = chunk
                skip = msadpcm_best_skip(chunk, s.channels, s.block_align)
                if skip > 0 and skip < len(chunk):
                    wav_chunk = chunk[skip:]
                if s.block_align > 0:
                    wav_chunk = wav_chunk[: (len(wav_chunk) // s.block_align) * s.block_align]

                ns = s.num_samples
                if ns <= 0 and s.block_align > 0:
                    spb = msadpcm_samples_per_block(s.block_align, s.channels)
                    ns = (len(wav_chunk) // s.block_align) * spb

                if DECODE_MSADPCM_TO_PCM_WAV:
                    pcm = decode_msadpcm_to_pcm16(wav_chunk, s.sample_rate, s.channels, s.block_align, ns)
                    wav_bytes = build_pcm16_wav(pcm, s.sample_rate, s.channels)
                else:
                    wav_bytes = build_msadpcm_wav(wav_chunk, s.sample_rate, s.channels, s.block_align, ns)
            elif s.codec == 0x00:
                pcm_norm, pcm_layout = normalize_pcm16_for_wav(chunk, s.channels, mode=getattr(s, 'pcm_layout', 'auto'))
                s.pcm_layout = pcm_layout
                wav_bytes = build_pcm16_wav(pcm_norm, s.sample_rate, s.channels)
            elif s.codec == 0x90:
                # NGC DSP (DSP_HEAD): decode to PCM16 and write WAV
                if s.channels != 1:
                    raise NotImplementedError('DSP_HEAD with channels>1 not supported yet')
                if s.dsp_header_pos < 0:
                    raise ValueError('DSP subsong missing dsp header position')
                coefs_list, hist_list = read_dsp_header_from_wbh(wbh_blob, s.dsp_header_pos, 1, bool(s.dsp_big_endian))
                pcm = decode_dsp_adpcm_mono(chunk, s.num_samples if s.num_samples > 0 else (len(chunk)//8)*14, coefs_list[0], hist_list[0][0], hist_list[0][1])
                wav_bytes = build_pcm16_wav(pcm, s.sample_rate, 1)
            else:
                raise ValueError(f"Unsupported codec 0x{s.codec:02X} for WAV export.")

            with open(os.path.join(sub_dir, wav_fn), "wb") as wf:
                wf.write(wav_bytes)
            wav_error = None
        except Exception as e:
            wav_fn = None
            wav_error = f"{type(e).__name__}: {e}"

        manifest.append({
            "index": s.index,
            "sound_index": s.sound_index,
            "subsound_index": s.subsound_index,
            "wbh_stream_off_pos": s.wbh_stream_off_pos,
            "wbh_stream_size_pos": s.wbh_stream_size_pos,
            "wbh_num_samples_pos": s.wbh_num_samples_pos,
            "wbd_body_offset": s.wbd_base_offset,
            "wbd_file_offset": s.wbd_file_offset,
            "orig_stream_offset": s.stream_offset,
            "orig_stream_size": s.stream_size,
            "sample_rate": s.sample_rate,
            "codec": s.codec,
            "dsp_header_pos": getattr(s, "dsp_header_pos", -1),
            "dsp_big_endian": getattr(s, "dsp_big_endian", False),
            "pcm_layout": getattr(s, "pcm_layout", "auto"),
            "channels": s.channels,
            "block_align": s.block_align,
            "num_samples": s.num_samples,
            "filename": fn,
            "wav_filename": wav_fn,
            "wav_error": wav_error,
        })

    with open(os.path.join(sub_dir, "subsongs.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return len(subsongs), f"Extracted {len(subsongs)} subsong(s) -> {os.path.basename(sub_dir)}"



def rebuild_wbd_and_rewrite_wbh(wbh_blob: bytes, wbd_blob: bytes, subsong_folder: str) -> tuple[bytes, bytes, str]:
    """
    Rebuild WBD payload from numbered subsong WAV files (PCM editing intermediate) in subsong_folder,
    re-encoding back into each subsong's original codec/layout as described in subsongs.json,
    and rewrite WBH offset/size/num_samples fields

    Supported codecs for repack:
      codec 0x00: PCM16LE (writes raw interleaved PCM16 into WBD)
      codec 0x10: Microsoft 4-bit ADPCM (encodes PCM -> MSADPCM blocks)
      codec 0x90: NGC DSP ADPCM (DSP_HEAD) (encodes PCM -> DSP ADPCM using existing coef table, mono supported)

    Notes:
      The tool's extracted WAVs are PCM16 for compatibility, repack expects replacement WAVs to be 16-bit PCM
      Replacement audio is auto-resampled to the original sample_rate and channel count
    """
    if not os.path.isdir(subsong_folder):
        return wbh_blob, wbd_blob, "No subsong folder found; using original WBD/WBH."

    manifest_path = os.path.join(subsong_folder, "subsongs.json")
    if not os.path.isfile(manifest_path):
        return wbh_blob, wbd_blob, "subsongs.json not found; cannot safely rewrite WBH."

    try:
        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    except Exception as e:
        return wbh_blob, wbd_blob, f"Failed to read subsongs.json: {e}"

    items = sorted(manifest, key=lambda d: int(d["index"]))
    new_wbh = bytearray(wbh_blob)

    # Prepare new chunks list (encoded bytes) and updated sample counts
    chunks: list[bytes] = []
    new_sample_counts: list[int | None] = []

    # Cache DSP coef tables per (pos, channels, endian)
    dsp_cache = {}

    for it in items:
        wav_fn = it.get("wav_filename")
        bin_fn = it.get("filename")
        codec = int(it.get("codec", 0x10))
        exp_sr = int(it.get("sample_rate", 0))
        exp_ch = int(it.get("channels", 1))
        exp_ba = int(it.get("block_align", 0))
        dsp_header_pos = int(it.get("dsp_header_pos", -1))
        dsp_big_endian = bool(it.get("dsp_big_endian", False))

        # load replacement WAV (preferred) or fall back to raw bin
        pcm_samples = None
        pcm_num = None
        if wav_fn:
            wp = os.path.join(subsong_folder, wav_fn)
            if os.path.isfile(wp):
                try:
                    info = read_pcm_wav_int16(wp)
                    pcm_samples = info["samples"]
                    pcm_num = int(info["num_samples"])
                    src_sr = int(info["sample_rate"])
                    src_ch = int(info["channels"])

                    # normalize channels then resample
                    pcm_samples = _mix_channels(pcm_samples, exp_ch)
                    pcm_samples = _resample_linear(pcm_samples, src_sr, exp_sr)
                    pcm_num = len(pcm_samples[0]) if pcm_samples else 0
                except Exception as e:
                    return wbh_blob, wbd_blob, f"Replacement WAV '{wav_fn}' must be 16-bit PCM. Error: {e}"

        # encode or fallback to .bin if WAV missing
        encoded = None
        new_ns = None

        if pcm_samples is not None:
            # encode based on expected codec
            try:
                if codec == 0x00:
                    encoded = _samples_to_pcm_bytes(pcm_samples)
                    new_ns = len(pcm_samples[0]) if pcm_samples else 0

                elif codec == 0x10:
                    if exp_ba <= 0:
                        return wbh_blob, wbd_blob, f"Missing/invalid block_align for MSADPCM subsong index {it.get('index')}."
                    encoded, new_ns = encode_msadpcm(pcm_samples, exp_sr, exp_ch, exp_ba)

                elif codec == 0x90:
                    # DSP_HEAD: encode mono only for now
                    if exp_ch != 1:
                        return wbh_blob, wbd_blob, f"DSP_HEAD repack currently supports mono only (subsong {it.get('index')} has {exp_ch}ch)."
                    if dsp_header_pos < 0:
                        return wbh_blob, wbd_blob, f"Missing dsp_header_pos for DSP subsong index {it.get('index')}."
                    key = (dsp_header_pos, exp_ch, dsp_big_endian)
                    if key not in dsp_cache:
                        coefs_all, hist_all = read_dsp_header_from_wbh(new_wbh, dsp_header_pos, exp_ch, dsp_big_endian)
                        dsp_cache[key] = (coefs_all, hist_all)
                    coefs_all, hist_all = dsp_cache[key]
                    coefs = coefs_all[0]
                    # set initial history in WBH header to 0,0 for consistent encode/decode
                    # hist positions are at dsp_header_pos + 0x40 and +0x42 (per channel stride 0x60)
                    new_wbh[dsp_header_pos + 0x40 : dsp_header_pos + 0x42] = struct.pack(">h" if dsp_big_endian else "<h", 0)
                    new_wbh[dsp_header_pos + 0x42 : dsp_header_pos + 0x44] = struct.pack(">h" if dsp_big_endian else "<h", 0)

                    new_ns = len(pcm_samples[0]) if pcm_samples else 0
                    encoded = encode_dsp_adpcm_mono(pcm_samples[0], coefs, new_ns)

                else:
                    return wbh_blob, wbd_blob, f"Unsupported codec 0x{codec:02X} for repack (subsong {it.get('index')})."

            except Exception as e:
                return wbh_blob, wbd_blob, f"Failed to encode subsong {it.get('index')}: {e}"

        if encoded is None:
            # fallback to raw .bin
            if not bin_fn:
                return wbh_blob, wbd_blob, "subsongs.json missing filename entries."
            bp = os.path.join(subsong_folder, bin_fn)
            if not os.path.isfile(bp):
                return wbh_blob, wbd_blob, f"Missing subsong file: {bin_fn}"
            encoded = open(bp, "rb").read()
            new_ns = int(it.get("num_samples")) if it.get("num_samples") is not None else None

        chunks.append(encoded)
        new_sample_counts.append(int(new_ns) if new_ns is not None else None)

    # build new WBD payload sequentially
    new_payload = b"".join(chunks)

    # preserve original WBD header (first 0x0C) but replace payload after 0x0C
    if len(wbd_blob) < 0x0C:
        return wbh_blob, wbd_blob, "WBD too small."
    new_wbd = bytearray()
    new_wbd += wbd_blob[:0x0C]
    new_wbd += new_payload

    # rewrite WBH stream offsets/sizes and num_samples
    cursor = 0
    for it, chunk, sc in zip(items, chunks, new_sample_counts):
        off_pos = int(it["wbh_stream_off_pos"])
        size_pos = int(it["wbh_stream_size_pos"])
        ns_pos = int(it.get("wbh_num_samples_pos", -1))

        # WBH stores stream_offset relative to WBD payload start (WBD + 0x0C)
        new_wbh[off_pos:off_pos+4] = _p32le(cursor)
        new_wbh[size_pos:size_pos+4] = _p32le(len(chunk))
        if ns_pos >= 0 and sc is not None:
            new_wbh[ns_pos:ns_pos+4] = _p32le(int(sc))
        cursor += len(chunk)

    return bytes(new_wbh), bytes(new_wbd), f"Rebuilt WBD from {len(chunks)} subsong(s): WAV->original codec repack (PCM/MSADPCM/DSP)."



# wrapper unpack/pack

def try_unpack_wrapper_pairs(file_path: str, out_dir: str, base_offset: int = 0, max_pairs: int = 512, extract_subsongs: bool = True) -> ParseResult:
    """
    Parse one or more wrapper blocks sequentially
    For WBD: declared size may overrun EOF, if so read to EOF - 6 (taildata)
    """
    try:
        file_size = os.path.getsize(file_path)
    except OSError as e:
        return ParseResult(False, 0, f"Could not stat file: {e}")

    pairs = 0
    cursor = base_offset

    try:
        with open(file_path, "rb") as f:
            for _ in range(max_pairs):
                if cursor + 16 > file_size:
                    break

                f.seek(cursor)
                hdr = f.read(16)
                if len(hdr) != 16:
                    break

                wbh_off, wbh_size, wbd_off, wbd_size = struct.unpack("<4I", hdr)

                if wbh_off < 0x10 or wbh_size <= 0 or wbd_off < 0x10 or wbd_size <= 0:
                    break

                wbh_abs = cursor + wbh_off
                wbd_abs = cursor + wbd_off

                # WBH must fit in file
                if (wbh_abs + wbh_size) > file_size or (wbh_abs < 0) or (wbd_abs < 0):
                    break

                if (wbd_abs + 8) > file_size:
                    break

                # Validate signatures at declared offsets
                f.seek(wbh_abs)
                if f.read(8) != SIG_WBH:
                    break

                f.seek(wbd_abs)
                if f.read(8) != SIG_WBD:
                    break

                # Read WBH (exact)
                f.seek(wbh_abs)
                wbh_data = f.read(wbh_size)
                if len(wbh_data) != wbh_size:
                    break

                # Read WBD (declared if overruns EOF read to EOF - 6)
                declared_end = wbd_abs + wbd_size
                if declared_end > file_size:
                    read_end = max(wbd_abs, file_size - TAIL_LEN)
                else:
                    read_end = declared_end

                wbd_len = read_end - wbd_abs
                if wbd_len <= 0:
                    break

                f.seek(wbd_abs)
                wbd_data = f.read(wbd_len)
                if len(wbd_data) != wbd_len:
                    break

                stem = os.path.splitext(os.path.basename(file_path))[0]
                wbh_name = f"{stem}_{pairs:02d}.wbh"
                wbd_name = f"{stem}_{pairs:02d}.wbd"

                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, wbh_name), "wb") as out_wbh:
                    out_wbh.write(wbh_data)
                with open(os.path.join(out_dir, wbd_name), "wb") as out_wbd:
                    out_wbd.write(wbd_data)

                # Extract subsongs (optional)
                if extract_subsongs:
                    prefix = f"{stem}_{pairs:02d}"
                    extract_subsongs_to_folder(wbh_data, wbd_data, out_dir, prefix)

                pairs += 1

                # Advance cursor, clamp to EOF to avoid runaway on bogus sizes
                end_abs = cursor + max(wbh_off + wbh_size, wbd_off + wbd_size)
                cursor = min(end_abs, file_size)

                if cursor + 16 > file_size:
                    break

                # Peek next header, stop unless plausible
                f.seek(cursor)
                peek = f.read(16)
                if len(peek) != 16:
                    break

                p_wbh_off, p_wbh_size, p_wbd_off, p_wbd_size = struct.unpack("<4I", peek)
                plausible = (
                    p_wbh_off >= 0x10 and p_wbh_size > 0 and
                    p_wbd_off >= 0x10 and p_wbd_size > 0 and
                    (cursor + p_wbh_off + 8) <= file_size and
                    (cursor + p_wbd_off + 8) <= file_size
                )
                if not plausible:
                    break

    except OSError as e:
        return ParseResult(False, pairs, f"I/O error: {e}")
    except Exception as e:
        return ParseResult(False, pairs, f"Parse error: {e}")

    if pairs == 0:
        return ParseResult(False, 0, "Wrapper metadata found invalid at offset 0 (or signatures not at declared offsets).")
    return ParseResult(True, pairs, f"Extracted {pairs} WBH/WBD pair(s).")


def _collect_wb_pairs(wb_folder: str) -> list[tuple[int, str, str]]:
    files = os.listdir(wb_folder)
    wbh = [f for f in files if f.lower().endswith(".wbh")]
    wbd = [f for f in files if f.lower().endswith(".wbd")]

    def idx_from_name(fn: str) -> int:
        m = re.search(r"_(\d{2,4})\.(wbh|wbd)$", fn, flags=re.IGNORECASE)
        return int(m.group(1)) if m else 0

    wbh_map = {idx_from_name(fn): os.path.join(wb_folder, fn) for fn in wbh}
    wbd_map = {idx_from_name(fn): os.path.join(wb_folder, fn) for fn in wbd}

    common = sorted(set(wbh_map.keys()) & set(wbd_map.keys()))
    return [(i, wbh_map[i], wbd_map[i]) for i in common]


def build_wrapped_bin(original_bin: str, wb_folder: str, out_dir: str, tail_len: int = 6) -> tuple[int, list[str], list[str]]:
    """
    Build new wrapped .bin file(s) from a wb_folder containing .wbh/.wbd (+ optional subsong folders)
    Returns (count, list_of_output_paths, list_of_notes)
    """
    if not os.path.isfile(original_bin):
        raise FileNotFoundError(f"Original file not found: {original_bin}")

    # Read taildata from original (ALWAYS)
    orig_size = os.path.getsize(original_bin)
    if orig_size < tail_len:
        raise ValueError("Original file is too small to contain taildata.")
    with open(original_bin, "rb") as f:
        f.seek(orig_size - tail_len)
        tail = f.read(tail_len)
        if len(tail) != tail_len:
            raise ValueError("Failed to read taildata from original.")

    pairs = _collect_wb_pairs(wb_folder)
    if not pairs:
        raise ValueError("No .wbh/.wbd pairs found in WB folder.")

    stem = os.path.splitext(os.path.basename(original_bin))[0]
    os.makedirs(out_dir, exist_ok=True)

    outputs: list[str] = []
    notes: list[str] = []

    for idx, wbh_path, wbd_path in pairs:
        wbh = open(wbh_path, "rb").read()
        wbd = open(wbd_path, "rb").read()

        if len(wbh) < 12 or wbh[:8] != SIG_WBH:
            raise ValueError(f"WBH signature invalid: {wbh_path}")
        if len(wbd) < 12 or wbd[:8] != SIG_WBD:
            raise ValueError(f"WBD signature invalid: {wbd_path}")

        # If subsongs folder exists, rebuild WBD payload and rewrite WBH offsets/sizes
        prefix = os.path.splitext(os.path.basename(wbh_path))[0]  # e.g., entry_XXXX_00
        subsong_dir = os.path.join(wb_folder, f"{prefix}_subsongs")
        wbh2, wbd2, note = rebuild_wbd_and_rewrite_wbh(wbh, wbd, subsong_dir)
        notes.append(f"{prefix}: {note}")

        wbh = wbh2
        wbd = wbd2

        # Compute wrapper header
        wbh_off = 0x10
        wbh_size = len(wbh)
        wbd_off = wbh_off + wbh_size
        wbd_size = len(wbd)  # NOT including taildata

        header = struct.pack("<4I", wbh_off, wbh_size, wbd_off, wbd_size)

        # Patch internal size fields to match wrapper header sizes
        wbh_patched = _patch_internal_size(wbh, SIG_WBH, wbh_size)
        wbd_patched = _patch_internal_size(wbd, SIG_WBD, wbd_size)

        # Output filename
        if len(pairs) == 1:
            out_name = f"{stem}.bin"
        else:
            out_name = f"{stem}_{idx:02d}.bin"

        out_path = unique_path(os.path.join(out_dir, out_name))

        with open(out_path, "wb") as out:
            out.write(header)
            out.write(wbh_patched)
            out.write(wbd_patched)
            out.write(tail)

        outputs.append(out_path)

    return len(outputs), outputs, notes


# UI app

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kybernes Scanner, WBH/WBD Tool")
        self.geometry("920x640")

        self.work_q: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.log_text = None
        self.status_var = tk.StringVar(value="Idle.")
        self.progress_var = tk.DoubleVar(value=0.0)

        # Tab 1 vars
        self.folder_var = tk.StringVar()
        self.include_subfolders_var = tk.BooleanVar(value=True)
        self.scan_limit_var = tk.StringVar(value="")
        self.extract_subsongs_var = tk.BooleanVar(value=True)

        # Tab 2 vars
        self.wrap_root_var = tk.StringVar()
        self.wrap_include_subfolders_var = tk.BooleanVar(value=True)
        self.out_folder_name_var = tk.StringVar(value="New_WBS")

        self._build_tabs()
        self.after(60, self._poll_queue)

    def _build_tabs(self):
        pad = {"padx": 10, "pady": 6}

        # TAB 1
        tab1 = ttk.Frame(self.nb)
        self.nb.add(tab1, text="Scan/Unpack")

        top = ttk.Frame(tab1)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Folder:").pack(side="left")
        ttk.Entry(top, textvariable=self.folder_var).pack(side="left", fill="x", expand=True, padx=(8, 8))
        ttk.Button(top, text="Browse", command=self.choose_folder).pack(side="left")

        opts = ttk.Frame(tab1)
        opts.pack(fill="x", **pad)

        ttk.Checkbutton(opts, text="Include subfolders", variable=self.include_subfolders_var).pack(side="left")
        ttk.Checkbutton(opts, text="Extract subsongs using WBH (KWB2)", variable=self.extract_subsongs_var).pack(side="left", padx=(18, 0))

        ttk.Label(opts, text="Max scan bytes per file (blank = full file):").pack(side="left", padx=(18, 6))
        ttk.Entry(opts, textvariable=self.scan_limit_var, width=14).pack(side="left")

        btns = ttk.Frame(tab1)
        btns.pack(fill="x", **pad)

        self.start_btn = ttk.Button(btns, text="Start Scan/Unpack", command=self.start_scan)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_work, state="disabled")
        self.stop_btn.pack(side="left", padx=(10, 0))

        # TAB 2
        tab2 = ttk.Frame(self.nb)
        self.nb.add(tab2, text="Wrap/Create")

        wtop = ttk.Frame(tab2)
        wtop.pack(fill="x", **pad)

        ttk.Label(wtop, text="Root folder:").pack(side="left")
        ttk.Entry(wtop, textvariable=self.wrap_root_var).pack(side="left", fill="x", expand=True, padx=(8, 8))
        ttk.Button(wtop, text="Browse", command=self.choose_wrap_root).pack(side="left")

        wopts = ttk.Frame(tab2)
        wopts.pack(fill="x", **pad)

        ttk.Checkbutton(wopts, text="Include subfolders", variable=self.wrap_include_subfolders_var).pack(side="left")

        ttk.Label(wopts, text="Output folder name:").pack(side="left", padx=(18, 6))
        ttk.Entry(wopts, textvariable=self.out_folder_name_var, width=14).pack(side="left")

        wbtns = ttk.Frame(tab2)
        wbtns.pack(fill="x", **pad)

        self.wrap_start_btn = ttk.Button(wbtns, text="Start Wrap/Create", command=self.start_wrap)
        self.wrap_start_btn.pack(side="left")
        self.wrap_stop_btn = ttk.Button(wbtns, text="Stop", command=self.stop_work, state="disabled")
        self.wrap_stop_btn.pack(side="left", padx=(10, 0))

        # Shared log/progress
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=6)

        pbar = ttk.Progressbar(bottom, variable=self.progress_var, maximum=100.0)
        pbar.pack(side="left", fill="x", expand=True)
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", padx=(10, 0))

        ttk.Label(self, text="Log:").pack(anchor="w", padx=10, pady=(4, 0))

        log_frame = ttk.Frame(self)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scroll.set)

        self._log("Ready.")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    def _set_busy(self, busy: bool):
        if busy:
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.wrap_start_btn.config(state="disabled")
            self.wrap_stop_btn.config(state="normal")
        else:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.wrap_start_btn.config(state="normal")
            self.wrap_stop_btn.config(state="disabled")

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select a folder to scan")
        if folder:
            self.folder_var.set(folder)

    def choose_wrap_root(self):
        folder = filedialog.askdirectory(title="Select a root folder for wrapping")
        if folder:
            self.wrap_root_var.set(folder)

    def stop_work(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self._log("Stop requested")

    # tab 1 scan worker

    def start_scan(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "A task is already running.")
            return

        max_scan = None
        raw = self.scan_limit_var.get().strip()
        if raw:
            try:
                max_scan = int(raw, 0)
                if max_scan <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Max scan bytes must be a positive integer (decimal or hex like 0x200000).")
                return

        self.stop_event.clear()
        self.progress_var.set(0.0)
        self.status_var.set("Scanning")
        self._set_busy(True)

        self._log(f"Starting scan in: {folder}")
        self._log(f"Include subfolders: {self.include_subfolders_var.get()}")
        self._log(f"Extract subsongs: {self.extract_subsongs_var.get()}")
        self._log(f"Max scan bytes per file: {max_scan if max_scan is not None else 'FULL'}")

        self.worker_thread = threading.Thread(
            target=self._worker_scan,
            args=(folder, self.include_subfolders_var.get(), max_scan, self.extract_subsongs_var.get()),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker_scan(self, folder: str, include_subfolders: bool, max_scan_bytes: int | None, extract_subsongs: bool):
        file_list: list[str] = []
        try:
            if include_subfolders:
                for root, _, files in os.walk(folder):
                    if self.stop_event.is_set():
                        self.work_q.put(("done",))
                        return
                    for fn in files:
                        file_list.append(os.path.join(root, fn))
            else:
                for fn in os.listdir(folder):
                    if self.stop_event.is_set():
                        self.work_q.put(("done",))
                        return
                    p = os.path.join(folder, fn)
                    if os.path.isfile(p):
                        file_list.append(p)
        except Exception as e:
            self.work_q.put(("log", f"Error listing files: {e}"))
            self.work_q.put(("done",))
            return

        total = len(file_list)
        self.work_q.put(("log", f"Found {total} file(s) to scan."))
        if total == 0:
            self.work_q.put(("done",))
            return

        hits = 0
        extracted_total_pairs = 0

        for i, path in enumerate(sorted(file_list, key=lambda p: natural_sort_key(os.path.basename(p))), start=1):
            if self.stop_event.is_set():
                self.work_q.put(("log", "Stopped by user."))
                break

            pct = (i / total) * 100.0
            self.work_q.put(("progress", pct, f"{i}/{total}"))

            try:
                if os.path.getsize(path) < 32:
                    continue
            except OSError:
                continue

            found = stream_find_signatures(path, [SIG_WBH, SIG_WBD], max_scan_bytes=max_scan_bytes)
            if found[SIG_WBH] is None or found[SIG_WBD] is None:
                continue

            hits += 1
            base = os.path.splitext(os.path.basename(path))[0]
            self.work_q.put(("log", f"HIT: {base} (found _HBW0000 at {found[SIG_WBH]:#x}, _DBW0000 at {found[SIG_WBD]:#x})"))

            parent = os.path.dirname(path)
            out_dir = unique_path(os.path.join(parent, f"{base}_WB"))

            res = try_unpack_wrapper_pairs(path, out_dir, base_offset=0, extract_subsongs=extract_subsongs)
            if res.ok:
                extracted_total_pairs += res.pairs_extracted
                self.work_q.put(("log", f"  Unpacked -> {out_dir}  ({res.pairs_extracted} pair(s))"))
                if extract_subsongs:
                    self.work_q.put(("log", f"  Subsongs: see '*_subsongs' folders inside {os.path.basename(out_dir)}"))
            else:
                self.work_q.put(("log", f"  Unpack failed: {res.message}"))

        self.work_q.put(("log", f"Scan complete. Files with signatures: {hits}. Total pairs extracted: {extracted_total_pairs}."))
        self.work_q.put(("done",))

    # tab 2 wrap worker

    def start_wrap(self):
        root = self.wrap_root_var.get().strip()
        if not root or not os.path.isdir(root):
            messagebox.showerror("Error", "Please select a valid root folder.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "A task is already running.")
            return

        out_name = (self.out_folder_name_var.get().strip() or "New_WBS").strip()
        if not out_name:
            out_name = "New_WBS"

        self.stop_event.clear()
        self.progress_var.set(0.0)
        self.status_var.set("Wrapping")
        self._set_busy(True)

        self._log(f"Starting wrap/create in: {root}")
        self._log(f"Include subfolders: {self.wrap_include_subfolders_var.get()}")
        self._log(f"Output folder name: {out_name}")

        self.worker_thread = threading.Thread(
            target=self._worker_wrap,
            args=(root, self.wrap_include_subfolders_var.get(), out_name),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker_wrap(self, root: str, include_subfolders: bool, out_folder_name: str):
        wb_folders: list[str] = []
        try:
            if include_subfolders:
                for droot, dirs, _files in os.walk(root):
                    if self.stop_event.is_set():
                        self.work_q.put(("done",))
                        return
                    for d in dirs:
                        if "_WB" in d:
                            wb_folders.append(os.path.join(droot, d))
            else:
                for d in os.listdir(root):
                    p = os.path.join(root, d)
                    if os.path.isdir(p) and "_WB" in d:
                        wb_folders.append(p)
        except Exception as e:
            self.work_q.put(("log", f"Error scanning folders: {e}"))
            self.work_q.put(("done",))
            return

        wb_folders = sorted(wb_folders, key=lambda p: natural_sort_key(os.path.basename(p)))
        total = len(wb_folders)
        self.work_q.put(("log", f"Found {total} _WB folder(s)."))
        if total == 0:
            self.work_q.put(("done",))
            return

        out_dir = unique_path(os.path.join(root, out_folder_name))
        os.makedirs(out_dir, exist_ok=True)
        self.work_q.put(("log", f"Output folder: {out_dir}"))

        made_files = 0
        made_sets = 0
        skipped = 0

        for i, wb_folder in enumerate(wb_folders, start=1):
            if self.stop_event.is_set():
                self.work_q.put(("log", "Stopped by user."))
                break

            pct = (i / total) * 100.0
            self.work_q.put(("progress", pct, f"{i}/{total}"))

            folder_name = os.path.basename(wb_folder)
            stem = folder_name[:-3] if folder_name.endswith("_WB") else folder_name.replace("_WB", "")

            parent = os.path.dirname(wb_folder)

            original = os.path.join(parent, f"{stem}.bin")
            if not os.path.isfile(original):
                candidates = [os.path.join(parent, fn) for fn in os.listdir(parent)
                              if os.path.isfile(os.path.join(parent, fn)) and os.path.splitext(fn)[0] == stem]
                candidates = sorted(candidates, key=lambda p: natural_sort_key(os.path.basename(p)))
                original = candidates[0] if candidates else ""

            if not original or not os.path.isfile(original):
                skipped += 1
                self.work_q.put(("log", f"SKIP: {folder_name} -> no original file found for stem '{stem}' next to folder."))
                continue

            try:
                count, outputs, notes = build_wrapped_bin(original, wb_folder, out_dir, tail_len=TAIL_LEN)
                made_sets += 1
                made_files += count
                for note in notes:
                    self.work_q.put(("log", f"NOTE: {note}"))
                for op in outputs:
                    self.work_q.put(("log", f"OK: {os.path.basename(original)} + {folder_name} -> {os.path.basename(op)}"))
            except Exception as e:
                skipped += 1
                self.work_q.put(("log", f"FAIL: {folder_name} ({os.path.basename(original)}) -> {e}"))

        self.work_q.put(("log", f"Wrap complete. Sets processed: {made_sets}. New files: {made_files}. Skipped/failed: {skipped}."))
        self.work_q.put(("done",))

    # queue poll

    def _poll_queue(self):
        try:
            while True:
                item = self.work_q.get_nowait()
                tag = item[0]
                if tag == "log":
                    self._log(item[1])
                elif tag == "progress":
                    pct, frac = item[1], item[2]
                    self.progress_var.set(pct)
                    self.status_var.set(f"{pct:5.1f}%  ({frac})")
                elif tag == "done":
                    self.status_var.set("Idle.")
                    self.progress_var.set(0.0)
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(60, self._poll_queue)


if __name__ == "__main__":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    App().mainloop()
