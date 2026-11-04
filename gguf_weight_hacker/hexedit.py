"""Block-aware weight hex editor (backend).

Produces annotated hex views: every byte is tagged with the structure it belongs to — header,
KV metadata, tensor-info table, padding, or ``tensor:block:field`` for weight payloads — and rows
carry decoded values (fp16 scales, int8 quants, nibbles). Edits go through
:class:`gguf_weight_hacker.patcher.Patcher` so they inherit backup, journaling and header protection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inspector import GGUFReader, TensorInfo
from .quant_codec import dequantize_raw

REGION_HEADER, REGION_KV, REGION_INFO, REGION_PAD, REGION_TENSOR = "header", "kv", "tensor_info", "padding", "tensor"


def _tensor_sorted(reader: GGUFReader) -> list[TensorInfo]:
    return sorted(reader.tensors.values(), key=lambda t: t.offset_bytes)


def locate(reader: GGUFReader, offset: int) -> dict:
    """Describe what lives at an absolute file offset."""
    if offset < 24:
        return {"region": REGION_HEADER, "field": "magic/version/counts"}
    if offset < reader._tensor_info_off:
        return {"region": REGION_KV, "field": "kv"}
    if offset < reader._data_align_off:
        return {"region": REGION_INFO, "field": "tensor_info"}
    if offset < reader.data_start:
        return {"region": REGION_PAD, "field": "padding"}
    for t in _tensor_sorted(reader):
        if t.offset_bytes <= offset < t.offset_bytes + t.size_bytes:
            rel = offset - t.offset_bytes
            bi, bo = divmod(rel, t.block_size_bytes)
            fld = next((f for f in t.layout.fields if f.offset <= bo < f.offset + f.nbytes), None)
            return {"region": REGION_TENSOR, "tensor": t.name, "block": bi, "block_offset": bo,
                    "field": fld.name if fld else "?", "role": fld.role if fld else None}
    return {"region": REGION_PAD, "field": "padding"}


def tensor_start(reader: GGUFReader, name: str, block: int = 0) -> int:
    t = reader.tensors[name]
    return t.offset_bytes + block * t.block_size_bytes


def hex_view(reader: GGUFReader, offset: int, length: int = 256, width: int = 16) -> dict:
    """Annotated rows covering ``[offset, offset+length)``, clamped to the file."""
    end = min(reader.filesize(), offset + length)
    offset = max(0, min(offset, end))
    data = bytes(reader.mmap[offset:end])
    rows = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        tags = [locate(reader, offset + i + j) for j in range(len(chunk))]
        rows.append({
            "offset": offset + i,
            "hex": chunk.hex(),
            "ascii": "".join(chr(b) if 32 <= b < 127 else "." for b in chunk),
            "tags": [_tag_key(t) for t in tags],
            "roles": [t.get("role") or t["region"] for t in tags],
        })
    return {"offset": offset, "length": len(data), "width": width, "rows": rows,
            "here": locate(reader, offset), "file_size": reader.filesize(), "data_start": reader.data_start}


def _tag_key(t: dict) -> str:
    if t["region"] == REGION_TENSOR:
        return f'{t["tensor"]}#{t["block"]}.{t["field"]}'
    return t["region"]


def block_view(reader: GGUFReader, tensor: str, block: int) -> dict:
    """One quant block: byte range, per-field decode, and the dequantized elements."""
    t = reader.tensors[tensor]
    if not 0 <= block < t.blocks_count:
        raise IndexError(f"block {block} out of range 0..{t.blocks_count - 1}")
    off = tensor_start(reader, tensor, block)
    raw = np.frombuffer(reader.mmap, np.uint8, t.block_size_bytes, off)
    fields = []
    for f in t.layout.fields:
        seg = raw[f.offset:f.offset + f.nbytes]
        vals = seg.view(np.dtype(f.dtype)) if f.dtype not in ("u1",) else seg
        shown = vals.astype(np.float32).tolist() if f.dtype.endswith("f2") or f.dtype.endswith("f4") else vals.tolist()
        fields.append({"name": f.name, "role": f.role, "offset": off + f.offset, "block_offset": f.offset,
                       "nbytes": f.nbytes, "dtype": f.dtype, "values": shown[:64],
                       "truncated": len(shown) > 64})
    elems = dequantize_raw(raw.copy(), t.type, t.layout.block_elems).tolist() if not t.layout.is_float else None
    return {"tensor": tensor, "block": block, "offset": off, "size": t.block_size_bytes,
            "type": t.type.name, "fields": fields, "dequant": elems}


def hexdump_text(reader: GGUFReader, offset: int, length: int = 256, width: int = 16, color: bool = False) -> str:
    """Plain-text (optionally ANSI-colored) hexdump with structure annotations in the margin."""
    palette = {"scale": "33", "min": "36", "subscale": "35", "hbits": "34", "payload": "0",
               REGION_HEADER: "32", REGION_KV: "32", REGION_INFO: "32", REGION_PAD: "90"}
    view = hex_view(reader, offset, length, width)
    out = []
    for row in view["rows"]:
        cells = []
        for j in range(len(row["hex"]) // 2):
            h = row["hex"][2 * j:2 * j + 2]
            cells.append(f"\x1b[{palette.get(row['roles'][j], '0')}m{h}\x1b[0m" if color else h)
        pad = "   " * (width - len(cells))
        note = row["tags"][0]
        if len(set(row["tags"])) > 1:
            note += " …"
        out.append(f'{row["offset"]:010x}  {" ".join(cells)}{pad}  |{row["ascii"]:<{width}}|  {note}')
    return "\n".join(out)


def parse_hex(s: str) -> bytes:
    """Accept ``"de ad be ef"``, ``"0xdeadbeef"`` or ``"deadbeef"``."""
    s = s.strip().lower().replace("0x", "").replace(" ", "").replace(",", "")
    if len(s) % 2 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"invalid hex string: {s!r}")
    return bytes.fromhex(s)


def search_bytes(reader: GGUFReader, pattern: bytes, start: int = 0, limit: int = 50) -> list[int]:
    """Find occurrences of ``pattern`` in the file (used by the hex editor's find box)."""
    hits, pos = [], start
    while len(hits) < limit:
        i = reader.mmap.find(pattern, pos)
        if i < 0:
            break
        hits.append(i); pos = i + 1
    return hits
