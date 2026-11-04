"""Static Header & KV Inspector.

Memory-maps a GGUF file and parses the header, KV metadata and tensor info table without
touching tensor payloads. Each tensor is exposed as a zero-copy NumPy view over the mmap,
either as raw bytes or as a structured block array whose fields (``d``, ``dmin``,
``scales``, ``qs`` ...) map onto the on-disk C structs.
"""
from __future__ import annotations

import fnmatch
import hashlib
import mmap
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .constants import (GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, LLAMA_FTYPE_NAMES, SCALAR_FMT,
                        BlockLayout, GGMLType, GGUFValueType, layout_for)
from .log import get_logger

log = get_logger("inspector")

_LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)\.")


class GGUFFormatError(ValueError):
    pass


@dataclass
class KVEntry:
    key: str
    vtype: GGUFValueType
    value: Any
    offset: int
    item_type: GGUFValueType | None = None

    def summary(self, max_items: int = 8) -> Any:
        v = self.value
        if isinstance(v, list) and len(v) > max_items:
            return {"array_of": self.item_type.name if self.item_type else None,
                    "len": len(v), "head": [_jsonable(x) for x in v[:max_items]]}
        return _jsonable(v)


def _jsonable(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    if isinstance(v, float):
        if v != v: return "NaN"
        if v == float("inf"): return "Infinity"
        if v == float("-inf"): return "-Infinity"
    return v


@dataclass
class TensorInfo:
    """Descriptor for one tensor stored in the GGUF file."""
    name: str
    shape: tuple[int, ...]           # (dim0, dim1, ...) as stored (row-major, dim0 is fastest)
    type: GGMLType
    offset_bytes: int                # absolute byte offset in the file
    size_bytes: int
    blocks_count: int
    block_size_bytes: int
    layout: BlockLayout

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape: n *= d
        return n

    @property
    def layer_index(self) -> int | None:
        m = _LAYER_RE.search(self.name)
        return int(m.group(1)) if m else None

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.name,
            "shape": list(self.shape),
            "type": f"GGML_TYPE_{self.type.name}",
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "blocks_count": self.blocks_count,
            "block_size_bytes": self.block_size_bytes,
            "quant_params": {
                "block_elements": self.layout.block_elems,
                "has_subblock_scales": any(f.role == "subscale" for f in self.layout.fields),
            },
            "fields": [{"name": f.name, "offset": f.offset, "dtype": f.dtype,
                        "count": f.count, "role": f.role} for f in self.layout.fields],
            "layer": self.layer_index,
        }


class GGUFReader:
    """Memory-mapped GGUF reader.

    Opens the file read-only by default. Pass ``writable=True`` to obtain a read/write
    mapping suitable for in-place patching (see :mod:`gguf_weight_hacker.patcher`).
    """

    def __init__(self, path: str | os.PathLike, writable: bool = False):
        self.path = os.fspath(path)
        self._writable = writable
        self._file = open(self.path, "r+b" if writable else "rb")
        access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
        self._mm = mmap.mmap(self._file.fileno(), 0, access=access)
        self.alignment = GGUF_DEFAULT_ALIGNMENT

        self._parse_header()
        self._parse_kv()
        self._parse_tensor_infos()
        self._compute_tensor_offsets()

        log.info("opened %s version=%d tensors=%d kv=%d bytes=%d%s",
                 self.path, self.version, len(self.tensors), len(self.kv),
                 len(self._mm), " (rw)" if writable else "")

    # ---- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        try: self._mm.flush()
        except Exception: pass
        self._mm.close()
        self._file.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    @property
    def mmap(self) -> mmap.mmap:
        return self._mm

    @property
    def writable(self) -> bool:
        return self._writable

    def filesize(self) -> int:
        return len(self._mm)

    # ---- parsing primitives --------------------------------------------------
    def _read(self, fmt: str, off: int) -> tuple:
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self._mm, off), off + size

    def _read_u32(self, off: int) -> tuple[int, int]:
        (v,), off = self._read("<I", off); return v, off

    def _read_u64(self, off: int) -> tuple[int, int]:
        (v,), off = self._read("<Q", off); return v, off

    def _read_string(self, off: int) -> tuple[str, int]:
        (n,), off = self._read("<Q", off)
        data = bytes(self._mm[off:off + n]); off += n
        return data.decode("utf-8", "replace"), off

    def _read_value(self, vtype: GGUFValueType, off: int) -> tuple[Any, int]:
        if vtype == GGUFValueType.STRING:
            return self._read_string(off)
        if vtype == GGUFValueType.ARRAY:
            (it,), off = self._read("<I", off)
            (n,), off = self._read("<Q", off)
            items, item_type = [], GGUFValueType(it)
            for _ in range(n):
                v, off = self._read_value(item_type, off)
                items.append(v)
            return items, off
        fmt, _ = SCALAR_FMT[vtype]
        (v,), off = self._read("<" + fmt, off)
        return v, off

    def _item_type(self, entry_off: int) -> GGUFValueType | None:
        (_n,), p = self._read("<Q", entry_off)
        p += _n
        (vt,), p = self._read("<I", p)
        if GGUFValueType(vt) != GGUFValueType.ARRAY: return None
        (it,), _ = self._read("<I", p)
        return GGUFValueType(it)

    # ---- header, KV, tensor infos --------------------------------------------
    def _parse_header(self) -> None:
        (magic,), off = self._read("<I", 0)
        if magic != GGUF_MAGIC:
            raise GGUFFormatError(f"bad magic 0x{magic:08x}; not a GGUF file")
        (self.version,), off = self._read("<I", off)
        if self.version not in (2, 3):
            log.warning("GGUF v%d is unusual; proceeding", self.version)
        (self.tensor_count,), off = self._read("<Q", off)
        (self.kv_count,), off = self._read("<Q", off)
        self._kv_table_off = off

    def _parse_kv(self) -> None:
        self.kv: dict[str, KVEntry] = {}
        off = self._kv_table_off
        for _ in range(self.kv_count):
            entry_off = off
            key, off = self._read_string(off)
            (vt,), off = self._read("<I", off)
            vtype = GGUFValueType(vt)
            value, off = self._read_value(vtype, off)
            item_type = self._item_type(entry_off) if vtype == GGUFValueType.ARRAY else None
            self.kv[key] = KVEntry(key, vtype, value, entry_off, item_type)
        self._tensor_info_off = off
        if "general.alignment" in self.kv:
            self.alignment = int(self.kv["general.alignment"].value)

    def _parse_tensor_infos(self) -> None:
        self._infos_raw: list[tuple[str, tuple[int, ...], GGMLType, int]] = []
        off = self._tensor_info_off
        for _ in range(self.tensor_count):
            name, off = self._read_string(off)
            (n_dims,), off = self._read("<I", off)
            dims = []
            for _ in range(n_dims):
                (d,), off = self._read("<Q", off); dims.append(d)
            (t,), off = self._read("<I", off)
            (rel_off,), off = self._read("<Q", off)
            self._infos_raw.append((name, tuple(dims), GGMLType(t), rel_off))
        self._data_align_off = off

    def _align_up(self, x: int, a: int) -> int:
        return (x + a - 1) & ~(a - 1)

    def _compute_tensor_offsets(self) -> None:
        self.data_start = self._align_up(self._data_align_off, self.alignment)
        self.tensors: dict[str, TensorInfo] = {}
        for name, shape, t, rel in self._infos_raw:
            layout = layout_for(t)
            n_elems = 1
            for d in shape: n_elems *= d
            if n_elems % layout.block_elems:
                raise GGUFFormatError(f"{name}: {n_elems} elements not divisible by block {layout.block_elems}")
            blocks = n_elems // layout.block_elems
            size = blocks * layout.block_bytes
            self.tensors[name] = TensorInfo(
                name=name, shape=shape, type=t,
                offset_bytes=self.data_start + rel, size_bytes=size,
                blocks_count=blocks, block_size_bytes=layout.block_bytes, layout=layout,
            )

    # ---- lookup / listing ----------------------------------------------------
    def __contains__(self, name: str) -> bool: return name in self.tensors
    def __iter__(self) -> Iterator[TensorInfo]: return iter(self.tensors.values())
    def __len__(self) -> int: return len(self.tensors)
    def __getitem__(self, name: str) -> TensorInfo: return self.tensors[name]

    def find(self, pattern: str) -> list[TensorInfo]:
        """Shell-style glob over tensor names (``blk.14.*``, ``*attn_v*``)."""
        return [t for name, t in self.tensors.items() if fnmatch.fnmatchcase(name, pattern)]

    def layer_tensors(self, layer: int) -> list[TensorInfo]:
        return [t for t in self.tensors.values() if t.layer_index == layer]

    def num_layers(self) -> int:
        n = self.kv.get("llama.block_count") or self.kv.get("general.block_count")
        if n is not None: return int(n.value)
        seen = {t.layer_index for t in self.tensors.values() if t.layer_index is not None}
        return (max(seen) + 1) if seen else 0

    # ---- zero-copy tensor views ---------------------------------------------
    def tensor_bytes(self, name: str) -> np.ndarray:
        """1D ``uint8`` view straight over the mmap for this tensor's payload."""
        ti = self.tensors[name]
        arr = np.frombuffer(self._mm, dtype=np.uint8,
                            count=ti.size_bytes, offset=ti.offset_bytes)
        if not self._writable:
            arr.setflags(write=False)
        return arr

    def block_dtype(self, t: GGMLType | TensorInfo) -> np.dtype:
        """A structured ``np.dtype`` whose fields match the on-disk C struct."""
        layout = t.layout if isinstance(t, TensorInfo) else layout_for(t)
        names, formats, offsets = [], [], []
        for f in layout.fields:
            names.append(f.name)
            formats.append((f.dtype, f.count) if f.count > 1 else f.dtype)
            offsets.append(f.offset)
        return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                         "itemsize": layout.block_bytes})

    def blocks(self, name: str) -> np.ndarray:
        """Structured block array (zero-copy). Writable iff reader is writable."""
        ti = self.tensors[name]
        if ti.layout.block_elems == 1 and len(ti.layout.fields) == 1:
            return self.tensor_bytes(name).view(np.dtype(ti.layout.fields[0].dtype))
        dt = self.block_dtype(ti)
        arr = np.frombuffer(self._mm, dtype=dt, count=ti.blocks_count, offset=ti.offset_bytes)
        if not self._writable:
            arr.setflags(write=False)
        return arr

    def scales(self, name: str, field_name: str = "d") -> np.ndarray:
        """FP16 scale (or ``dmin``, ``m``) view as float32 for convenience. Read-only copy."""
        blocks = self.blocks(name)
        if field_name not in blocks.dtype.names:
            raise KeyError(f"{name} ({self.tensors[name].type.name}) has no field {field_name!r}")
        return blocks[field_name].astype(np.float32, copy=True).reshape(-1)

    # ---- meta ----------------------------------------------------------------
    def file_type_name(self) -> str:
        ft = self.kv.get("general.file_type")
        if ft is None: return "UNKNOWN"
        return LLAMA_FTYPE_NAMES.get(int(ft.value), f"FTYPE_{int(ft.value)}")

    def summary(self) -> dict:
        from collections import Counter
        type_counts = Counter(t.type.name for t in self.tensors.values())
        bytes_by_type = {}
        for t in self.tensors.values():
            bytes_by_type[t.type.name] = bytes_by_type.get(t.type.name, 0) + t.size_bytes
        arch = self.kv.get("general.architecture")
        return {
            "path": self.path,
            "size_bytes": self.filesize(),
            "version": self.version,
            "alignment": self.alignment,
            "architecture": arch.value if arch else None,
            "file_type": self.file_type_name(),
            "n_tensors": len(self.tensors),
            "n_layers": self.num_layers(),
            "data_start": self.data_start,
            "type_counts": dict(type_counts),
            "bytes_by_type": bytes_by_type,
            "kv_keys": list(self.kv.keys()),
        }

    def quick_checksum(self, name: str, sample: int = 16) -> str:
        """SHA-256 over a sample of blocks from a tensor (enough to detect edits)."""
        ti = self.tensors[name]
        raw = self.tensor_bytes(name)
        h = hashlib.sha256()
        n = ti.blocks_count
        sz = ti.block_size_bytes
        if n <= sample:
            h.update(bytes(raw))
        else:
            for i in np.linspace(0, n - 1, sample, dtype=np.int64):
                off = int(i) * sz
                h.update(bytes(raw[off:off + sz]))
        return h.hexdigest()


def open_gguf(path, writable: bool = False) -> GGUFReader:
    """Convenience constructor: ``with open_gguf("model.gguf") as r: ...``."""
    return GGUFReader(path, writable=writable)
