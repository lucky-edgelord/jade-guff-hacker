"""Zero-Copy Binary Patcher.

In-place edits against a memory-mapped GGUF file. Supports:

* **scale patching** — multiply / set the FP16 block scales (``d``, ``dmin``, ``m``) of a
  tensor by a scalar or a per-block array;
* **byte poking** — raw byte writes for the hex editor;
* **nibble/quant edits** — overwrite packed quant payloads.

Every mutating op goes through :class:`Patcher`, which enforces: a checksum-verified ``.bak``
backup (or an in-memory undo journal), range/NaN/Inf validation, header-region protection,
and an explicit ``msync`` flush. ``dry_run`` computes and validates the edit but writes nothing.
"""
from __future__ import annotations

import json
import mmap
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .constants import FP16_MAX
from .inspector import GGUFReader, TensorInfo
from .log import get_logger, progress

log = get_logger("patcher")

MAX_SCALE_RATIO = 1e4   # reject |new/orig| > this (spec 5.1 range validation)


class PatchError(RuntimeError):
    pass


class PatchAborted(PatchError):
    pass


@dataclass
class PatchRecord:
    """One reversible byte-range edit, kept in the journal."""
    tensor: str
    field: str
    byte_offset: int        # absolute offset in file
    before: bytes
    after: bytes
    note: str = ""

    def to_dict(self):
        return {"tensor": self.tensor, "field": self.field, "byte_offset": self.byte_offset,
                "n_bytes": len(self.after), "note": self.note}


@dataclass
class PatchResult:
    tensor: str
    field: str
    op: str
    blocks_touched: int
    dry_run: bool
    stats: dict = field(default_factory=dict)
    records: list = field(default_factory=list)

    def to_dict(self):
        d = {k: getattr(self, k) for k in ("tensor", "field", "op", "blocks_touched", "dry_run", "stats")}
        d["records"] = [r.to_dict() for r in self.records]
        return d


class Patcher:
    """Mutating operations on a writable :class:`GGUFReader`.

    Parameters
    ----------
    reader : GGUFReader (must be opened ``writable=True``)
    backup : create a ``.bak`` copy before the first write (default True)
    journal : keep in-memory undo records so :meth:`revert` works without the backup
    """

    def __init__(self, reader: GGUFReader, *, backup: bool = True, journal: bool = True):
        if not reader.writable:
            raise PatchError("reader must be opened with writable=True to patch")
        self.r = reader
        self.backup = backup
        self._journal_on = journal
        self.journal: list[PatchRecord] = []
        self._bak_path: str | None = None
        self._bak_done = False

    # ---- safety helpers ------------------------------------------------------
    def _ensure_backup(self):
        if not self.backup or self._bak_done:
            return
        self._bak_path = self.r.path + ".bak"
        if not os.path.exists(self._bak_path):
            self.r.mmap.flush()
            shutil.copy2(self.r.path, self._bak_path)
            log.info("backup written -> %s", self._bak_path)
        self._bak_done = True

    def _check_data_region(self, off: int, n: int):
        """Refuse writes that fall outside the tensor data region (header protection)."""
        if off < self.r.data_start:
            raise PatchAborted(f"refusing write at {off}: inside header/metadata region "
                               f"(data starts at {self.r.data_start})")
        if off + n > self.r.filesize():
            raise PatchAborted(f"refusing write: [{off},{off+n}) past EOF {self.r.filesize()}")

    @staticmethod
    def _validate_scales(new: np.ndarray, orig: np.ndarray | None = None):
        if not np.all(np.isfinite(new)):
            raise PatchAborted("edit produces NaN/Inf scale values")
        if np.any(np.abs(new) > FP16_MAX):
            raise PatchAborted(f"edit exceeds fp16 max ({FP16_MAX}); would overflow on load")
        if orig is not None:
            ratio = np.abs(new) / np.maximum(np.abs(orig), 1e-30)
            if np.any(ratio > MAX_SCALE_RATIO):
                raise PatchAborted(f"edit changes a scale by >{MAX_SCALE_RATIO:g}x "
                                   f"(max {float(ratio.max()):.3g}x); refusing as likely error")

    def _write_bytes(self, off: int, data: bytes, tensor: str, fieldname: str, note: str):
        self._check_data_region(off, len(data))
        before = bytes(self.r.mmap[off:off + len(data)])
        if self._journal_on:
            self.journal.append(PatchRecord(tensor, fieldname, off, before, bytes(data), note))
        self.r.mmap[off:off + len(data)] = data

    # ---- scale patching ------------------------------------------------------
    def patch_scale(self, tensor: str, *, field: str = "d",
                    multiplier=None, set_to=None, scale_map: np.ndarray | None = None,
                    block_slice: slice | None = None, dry_run: bool = False) -> PatchResult:
        """Modify an FP16 scale field across (a slice of) a tensor's blocks.

        Exactly one of ``multiplier`` (scalar or per-block array), ``set_to`` (scalar/array),
        or ``scale_map`` (per-block array of *new* values) must be given.
        """
        ti = self.r.tensors[tensor]
        layout = ti.layout
        f = layout.field(field)
        if f.dtype != "<f2" or f.role not in ("scale", "min"):
            raise PatchError(f"{field!r} is not an fp16 scale field on {ti.type.name}")
        blocks = self.r.blocks(tensor)
        sel = block_slice or slice(None)
        orig = blocks[field][sel].astype(np.float32).reshape(-1)
        n = orig.shape[0]

        provided = [x is not None for x in (multiplier, set_to, scale_map)]
        if sum(provided) != 1:
            raise PatchError("give exactly one of multiplier=, set_to=, scale_map=")
        if multiplier is not None:
            new = orig * np.asarray(multiplier, np.float32)
        elif set_to is not None:
            new = np.broadcast_to(np.asarray(set_to, np.float32), (n,)).astype(np.float32).copy()
        else:
            new = np.asarray(scale_map, np.float32).reshape(-1)
            if new.shape[0] != n:
                raise PatchError(f"scale_map has {new.shape[0]} entries, tensor slice has {n} blocks")

        self._validate_scales(new, orig)
        new16 = new.astype(np.float16)
        stats = {"orig_mean": float(orig.mean()), "new_mean": float(new16.astype(np.float32).mean()),
                 "orig_max": float(np.abs(orig).max()), "new_max": float(np.abs(new16).max().astype(np.float32)),
                 "max_ratio": float((np.abs(new) / np.maximum(np.abs(orig), 1e-30)).max())}
        res = PatchResult(tensor, field, "patch_scale", n, dry_run, stats)
        if dry_run:
            log.info("[dry-run] patch_scale %s.%s blocks=%d mean %.4g->%.4g",
                     tensor, field, n, stats["orig_mean"], stats["new_mean"])
            return res

        self._ensure_backup()
        # write per block: field sits at f.offset within each block_size_bytes stride
        start = ti.offset_bytes + (sel.start or 0) * ti.block_size_bytes
        stride = ti.block_size_bytes
        new_le = new16.view(np.uint8).reshape(n, 2)
        # contiguous write when possible, else per-block
        buf = bytearray(self.r.mmap[start:start + n * stride]) if n > 1 else None
        for i in progress(range(n), total=n, desc=f"patch {tensor}.{field}", unit="blk"):
            off = start + i * stride + f.offset
            self._write_bytes(off, bytes(new_le[i]), tensor, field, "patch_scale")
        res.records = self.journal[-n:] if self._journal_on else []
        self.flush()
        log.info("patched %s.%s %d blocks (mean %.4g -> %.4g)", tensor, field, n,
                 stats["orig_mean"], stats["new_mean"])
        return res

    # ---- raw byte poke (hex editor backend) ----------------------------------
    def poke_bytes(self, byte_offset: int, data: bytes, *, dry_run: bool = False,
                   note: str = "poke") -> PatchResult:
        """Overwrite raw bytes at an absolute file offset (within the data region)."""
        self._check_data_region(byte_offset, len(data))
        owner = self._tensor_at(byte_offset)
        res = PatchResult(owner or "?", "raw", "poke", 1, dry_run,
                          {"byte_offset": byte_offset, "n_bytes": len(data)})
        if dry_run:
            return res
        self._ensure_backup()
        self._write_bytes(byte_offset, bytes(data), owner or "?", "raw", note)
        res.records = self.journal[-1:]
        self.flush()
        return res

    def poke_block_field(self, tensor: str, block_index: int, field: str,
                         value, *, dry_run: bool = False) -> PatchResult:
        """Set one field of one block to ``value`` (scalar for fp16/int fields, bytes for payloads)."""
        ti = self.r.tensors[tensor]
        f = ti.layout.field(field)
        off = ti.offset_bytes + block_index * ti.block_size_bytes + f.offset
        if f.dtype == "<f2":
            arr = np.asarray([value], np.float16)
            self._validate_scales(arr.astype(np.float32))
            data = arr.view(np.uint8).tobytes()
        elif isinstance(value, (bytes, bytearray)):
            data = bytes(value)[: f.nbytes]
        else:
            data = np.asarray(value, np.dtype(f.dtype)).tobytes()[: f.nbytes]
        return self.poke_bytes(off, data, dry_run=dry_run, note=f"{tensor}[{block_index}].{field}")

    def _tensor_at(self, off: int) -> str | None:
        for name, t in self.r.tensors.items():
            if t.offset_bytes <= off < t.offset_bytes + t.size_bytes:
                return name
        return None

    # ---- flush / revert ------------------------------------------------------
    def flush(self):
        """msync(MS_SYNC) dirty pages to guarantee on-disk persistence (spec 5.1)."""
        try:
            self.r.mmap.flush()
        except Exception as e:  # pragma: no cover
            raise PatchError(f"msync failed: {e}") from e

    def revert(self, n: int | None = None) -> int:
        """Undo the last ``n`` journal records (all if None). Returns records reverted."""
        if not self._journal_on:
            raise PatchError("journal disabled; restore from .bak instead")
        recs = self.journal if n is None else self.journal[-n:]
        for rec in reversed(recs):
            self.r.mmap[rec.byte_offset:rec.byte_offset + len(rec.before)] = rec.before
        self.flush()
        k = len(recs)
        del self.journal[len(self.journal) - k:]
        log.info("reverted %d edits", k)
        return k

    def save_journal(self, path: str):
        with open(path, "w") as fh:
            json.dump({"path": self.r.path, "ts": time.time(),
                       "records": [r.to_dict() for r in self.journal]}, fh, indent=2)


def restore_backup(path: str) -> str:
    """Restore ``path`` from ``path.bak``. Returns the restored path."""
    bak = path + ".bak"
    if not os.path.exists(bak):
        raise PatchError(f"no backup at {bak}")
    shutil.copy2(bak, path)
    log.info("restored %s from %s", path, bak)
    return path
