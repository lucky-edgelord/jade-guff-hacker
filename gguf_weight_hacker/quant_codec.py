"""Quantization Codec.

Vectorized NumPy dequantizers for the common GGML block types, plus scale extraction.
Native paths (verified against the reference ``gguf`` library in the test-suite):
F32/F16/BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K. Any other type is routed to the
reference ``gguf.quants.dequantize`` backend when that library is importable, so coverage is
complete while the hot paths stay dependency-free.
"""
from __future__ import annotations

import numpy as np

from .constants import GGMLType, layout_for
from .log import get_logger

log = get_logger("quant_codec")

_NATIVE: dict[GGMLType, callable] = {}


def _native(t):
    def deco(fn): _NATIVE[t] = fn; return fn
    return deco


def _f16_view(buf_u8: np.ndarray, byte_off: int, count: int, nblocks: int, stride: int) -> np.ndarray:
    """Read ``count`` fp16 values per block starting at ``byte_off`` -> (nblocks, count) f32."""
    idx = (np.arange(nblocks)[:, None] * stride + byte_off + np.arange(count)[None, :] * 2)
    lo = buf_u8[idx].astype(np.uint16)
    hi = buf_u8[idx + 1].astype(np.uint16)
    return (lo | (hi << 8)).view(np.float16).astype(np.float32)


def _raw_blocks(raw: np.ndarray, nblocks: int, block_bytes: int) -> np.ndarray:
    return raw[: nblocks * block_bytes].reshape(nblocks, block_bytes)


# --------------------------------------------------------------------------- floats
@_native(GGMLType.F32)
def _deq_f32(raw, nblocks, layout):
    return raw.view(np.float32).astype(np.float32).reshape(nblocks, 1)


@_native(GGMLType.F16)
def _deq_f16(raw, nblocks, layout):
    return raw.view(np.float16).astype(np.float32).reshape(nblocks, 1)


@_native(GGMLType.BF16)
def _deq_bf16(raw, nblocks, layout):
    u16 = raw.view(np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32).reshape(nblocks, 1)


# --------------------------------------------------------------------------- legacy
def _nibbles(blocks_u8):
    lo = (blocks_u8 & 0x0F).astype(np.int32)
    hi = (blocks_u8 >> 4).astype(np.int32)
    return lo, hi


@_native(GGMLType.Q4_0)
def _deq_q4_0(raw, nblocks, layout):
    b = _raw_blocks(raw, nblocks, 18)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)        # (n,1)
    lo, hi = _nibbles(b[:, 2:18])
    out = np.empty((nblocks, 32), np.float32)
    out[:, 0:16] = (lo - 8) * d
    out[:, 16:32] = (hi - 8) * d
    return out


@_native(GGMLType.Q4_1)
def _deq_q4_1(raw, nblocks, layout):
    b = _raw_blocks(raw, nblocks, 20)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    m = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    lo, hi = _nibbles(b[:, 4:20])
    out = np.empty((nblocks, 32), np.float32)
    out[:, 0:16] = lo * d + m
    out[:, 16:32] = hi * d + m
    return out


def _q5_high(qh_u32, nblocks):
    j = np.arange(16)
    xh0 = (((qh_u32[:, None] >> j) << 4) & 0x10).astype(np.int32)
    xh1 = (((qh_u32[:, None] >> (j + 12))) & 0x10).astype(np.int32)
    return xh0, xh1


@_native(GGMLType.Q5_0)
def _deq_q5_0(raw, nblocks, layout):
    b = _raw_blocks(raw, nblocks, 22)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    qh = b[:, 2:6].copy().view(np.uint32).reshape(-1)
    lo, hi = _nibbles(b[:, 6:22])
    xh0, xh1 = _q5_high(qh, nblocks)
    out = np.empty((nblocks, 32), np.float32)
    out[:, 0:16] = ((lo | xh0) - 16) * d
    out[:, 16:32] = ((hi | xh1) - 16) * d
    return out


@_native(GGMLType.Q5_1)
def _deq_q5_1(raw, nblocks, layout):
    b = _raw_blocks(raw, nblocks, 24)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    m = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    qh = b[:, 4:8].copy().view(np.uint32).reshape(-1)
    lo, hi = _nibbles(b[:, 8:24])
    xh0, xh1 = _q5_high(qh, nblocks)
    out = np.empty((nblocks, 32), np.float32)
    out[:, 0:16] = (lo | xh0) * d + m
    out[:, 16:32] = (hi | xh1) * d + m
    return out


@_native(GGMLType.Q8_0)
def _deq_q8_0(raw, nblocks, layout):
    b = _raw_blocks(raw, nblocks, 34)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:34].view(np.int8).astype(np.float32)
    return qs * d


# --------------------------------------------------------------------------- K-quants
def _get_scale_min_k4(scales_u8):
    """Unpack Q4_K/Q5_K 6-bit (scale, min) pairs. scales_u8: (n,12) -> (sc(n,8), m(n,8))."""
    q = scales_u8.astype(np.uint32)
    sc = np.empty((q.shape[0], 8), np.uint32)
    m = np.empty((q.shape[0], 8), np.uint32)
    for j in range(8):
        if j < 4:
            sc[:, j] = q[:, j] & 63
            m[:, j] = q[:, j + 4] & 63
        else:
            sc[:, j] = (q[:, j + 4] & 0x0F) | ((q[:, j - 4] >> 6) << 4)
            m[:, j] = (q[:, j + 4] >> 4) | ((q[:, j - 0] >> 6) << 4)
    return sc.astype(np.float32), m.astype(np.float32)


@_native(GGMLType.Q4_K)
def _deq_q4_k(raw, nblocks, layout):
    b = _raw_blocks(raw, nblocks, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)         # (n,1)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mn = _get_scale_min_k4(b[:, 4:16])                           # (n,8)
    qs = b[:, 16:144]                                                # (n,128)
    out = np.empty((nblocks, 256), np.float32)
    for g in range(4):                                              # 4 groups of 64
        chunk = qs[:, g * 32:(g + 1) * 32]
        lo = (chunk & 0x0F).astype(np.float32)
        hi = (chunk >> 4).astype(np.float32)
        is0, is1 = 2 * g, 2 * g + 1
        d1 = (d * sc[:, is0:is0 + 1]); m1 = (dmin * mn[:, is0:is0 + 1])
        d2 = (d * sc[:, is1:is1 + 1]); m2 = (dmin * mn[:, is1:is1 + 1])
        out[:, g * 64:g * 64 + 32] = d1 * lo - m1
        out[:, g * 64 + 32:g * 64 + 64] = d2 * hi - m2
    return out


@_native(GGMLType.Q6_K)
def _deq_q6_k(raw, nblocks, layout):
    b = _raw_blocks(raw, nblocks, 210)
    ql = b[:, 0:128].astype(np.int32)
    qh = b[:, 128:192].astype(np.int32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)             # (n,16)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)    # (n,1)
    out = np.empty((nblocks, 256), np.float32)
    for n in range(2):                                             # two halves of 128
        qlh = ql[:, n * 64:(n + 1) * 64]
        qhh = qh[:, n * 32:(n + 1) * 32]
        sch = sc[:, n * 8:(n + 1) * 8]
        l = np.arange(32)
        is_ = l // 16
        q1 = ((qlh[:, l] & 0x0F) | (((qhh >> 0) & 3) << 4)) - 32
        q2 = ((qlh[:, l + 32] & 0x0F) | (((qhh >> 2) & 3) << 4)) - 32
        q3 = ((qlh[:, l] >> 4) | (((qhh >> 4) & 3) << 4)) - 32
        q4 = ((qlh[:, l + 32] >> 4) | (((qhh >> 6) & 3) << 4)) - 32
        base = n * 128
        out[:, base + 0:base + 32] = d * sch[:, is_ + 0] * q1
        out[:, base + 32:base + 64] = d * sch[:, is_ + 2] * q2
        out[:, base + 64:base + 96] = d * sch[:, is_ + 4] * q3
        out[:, base + 96:base + 128] = d * sch[:, is_ + 6] * q4
    return out


# --------------------------------------------------------------------------- public API
def native_types() -> set[GGMLType]:
    return set(_NATIVE)


def has_reference() -> bool:
    try:
        import gguf.quants  # noqa: F401
        return True
    except Exception:
        return False


def dequantize_raw(raw: np.ndarray, t: GGMLType, n_elements: int) -> np.ndarray:
    """Dequantize a flat ``uint8`` payload to a 1D float32 array of ``n_elements``."""
    t = GGMLType(t) if not isinstance(t, GGMLType) else t
    layout = layout_for(t)
    nblocks = n_elements // layout.block_elems
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if t in _NATIVE:
        out = _NATIVE[t](raw, nblocks, layout)
        return out.reshape(-1)[:n_elements]
    # fallback to reference implementation
    try:
        from gguf.quants import dequantize as _ref
    except Exception as e:  # pragma: no cover
        raise NotImplementedError(
            f"No native dequantizer for {t.name} and reference 'gguf' not available ({e})"
        ) from None
    out = _ref(raw[: nblocks * layout.block_bytes].copy(), t)
    return np.asarray(out, np.float32).reshape(-1)[:n_elements]


def dequantize_tensor(reader, name: str, rows: int | None = None) -> np.ndarray:
    """Dequantize a whole tensor to float32, reshaped to its 2D ``(d1, d0)`` logical shape."""
    ti = reader.tensors[name]
    raw = np.asarray(reader.tensor_bytes(name))
    flat = dequantize_raw(raw, ti.type, ti.n_elements)
    shape = ti.shape
    if len(shape) >= 2:
        d0 = shape[0]
        return flat.reshape(-1, d0)[:rows] if rows else flat.reshape(-1, d0)
    return flat


def extract_scales(reader, name: str, field: str = "d") -> np.ndarray:
    """Return the per-block scale array (float32) for the given scale field."""
    return reader.scales(name, field)


def block_stats(reader, name: str) -> dict:
    """Cheap per-tensor statistics derived from scales (no full dequant)."""
    ti = reader.tensors[name]
    out = {"tensor": name, "type": ti.type.name, "blocks": ti.blocks_count,
           "elements": ti.n_elements, "shape": list(ti.shape)}
    layout = ti.layout
    for f in layout.scale_fields:
        s = reader.scales(name, f.name)
        finite = s[np.isfinite(s)]
        out[f.name] = {
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
            "n_nan": int(np.isnan(s).sum()), "n_inf": int(np.isinf(s).sum()),
        }
    return out
