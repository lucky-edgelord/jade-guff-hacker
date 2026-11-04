import numpy as np
import pytest

from gguf_weight_hacker import open_gguf, dequantize_tensor
from gguf_weight_hacker.constants import BLOCK_LAYOUTS, GGMLType, layout_for
from gguf_weight_hacker.hexedit import block_view, hex_view, locate, parse_hex, tensor_start
from gguf_weight_hacker.patcher import PatchAborted, PatchError, Patcher, restore_backup
from gguf_weight_hacker.quant_codec import dequantize_raw, native_types

gguf = pytest.importorskip("gguf")
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as RT  # noqa: E402
from gguf.quants import dequantize as ref_dequant  # noqa: E402


# ---- layouts / codec ----------------------------------------------------------------
def test_layouts_match_reference_sizes():
    for t, lay in BLOCK_LAYOUTS.items():
        if t.name not in RT.__members__:
            continue
        be, bb = GGML_QUANT_SIZES[RT[t.name]]
        assert (lay.block_elems, lay.block_bytes) == (be, bb), t.name
        # fields must tile the block without overlap or overrun
        spans = sorted((f.offset, f.offset + f.nbytes) for f in lay.fields)
        assert spans[-1][1] <= lay.block_bytes, t.name
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0, f"{t.name} fields overlap"


@pytest.mark.parametrize("t", sorted([t for t in native_types() if t.name not in ("F32", "F16", "BF16")], key=int))
def test_dequant_matches_reference(t):
    rng = np.random.default_rng(1)
    lay = layout_for(t); n = 6
    raw = rng.integers(0, 256, size=(n, lay.block_bytes), dtype=np.uint8)
    for f in lay.fields:  # sane fp16 scales so results are finite
        if f.dtype == "<f2" and f.role in ("scale", "min"):
            raw[:, f.offset:f.offset + 2] = (rng.random((n, 1)) * 0.05 + 0.01).astype(np.float16).view(np.uint8)
    raw = raw.reshape(-1)
    mine = dequantize_raw(raw.copy(), t, n * lay.block_elems)
    ref = np.asarray(ref_dequant(raw.copy(), RT[t.name]), np.float32).reshape(-1)
    np.testing.assert_allclose(mine, ref, rtol=1e-6, atol=1e-6)


# ---- inspector ----------------------------------------------------------------------
def test_inspector_matches_reference_reader(model):
    ref = gguf.GGUFReader(model)
    with open_gguf(model) as r:
        assert len(r.tensors) == len(ref.tensors)
        for rt in ref.tensors:
            ti = r.tensors[rt.name]
            assert ti.offset_bytes == rt.data_offset, rt.name
            assert ti.size_bytes == rt.n_bytes, rt.name
            assert ti.type.value == int(rt.tensor_type), rt.name
        assert r.kv["general.architecture"].value == "llama"
        assert r.summary()["n_layers"] == 4


def test_bad_magic(tmp_path):
    from gguf_weight_hacker import GGUFFormatError
    p = tmp_path / "x.gguf"; p.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(GGUFFormatError):
        open_gguf(str(p))


def test_struct_view_is_zero_copy(model):
    with open_gguf(model) as r:
        b = r.blocks("blk.0.attn_q.weight")
        assert b.dtype.names == ("d", "qs")
        assert not b.flags.writeable
        s = r.scales("blk.0.attn_q.weight")
        assert s.shape == (r.tensors["blk.0.attn_q.weight"].blocks_count,)


# ---- patcher ------------------------------------------------------------------------
T = "blk.1.ffn_down.weight"


def test_patch_scale_and_revert(model):
    with open_gguf(model) as r:
        before = r.scales(T).copy(); cs = r.quick_checksum(T)
    with open_gguf(model, writable=True) as r:
        p = Patcher(r)
        p.patch_scale(T, multiplier=2.0)
        np.testing.assert_allclose(r.scales(T), before * 2, rtol=1e-3)
        p.revert()
    with open_gguf(model) as r:
        assert r.quick_checksum(T) == cs


def test_dry_run_writes_nothing(model):
    import hashlib
    h0 = hashlib.md5(open(model, "rb").read()).hexdigest()
    with open_gguf(model, writable=True) as r:
        Patcher(r).patch_scale(T, multiplier=3.0, dry_run=True)
    assert hashlib.md5(open(model, "rb").read()).hexdigest() == h0


@pytest.mark.parametrize("kw", [{"multiplier": 1e9}, {"multiplier": float("nan")}, {"set_to": 1e6}, {"multiplier": float("inf")}])
def test_range_validation(model, kw):
    with open_gguf(model, writable=True) as r:
        with pytest.raises(PatchAborted):
            Patcher(r, backup=False).patch_scale(T, **kw)


def test_requires_exactly_one_edit(model):
    with open_gguf(model, writable=True) as r:
        p = Patcher(r, backup=False)
        with pytest.raises(PatchError):
            p.patch_scale(T)
        with pytest.raises(PatchError):
            p.patch_scale(T, multiplier=1.0, set_to=1.0)


def test_header_is_protected(model):
    with open_gguf(model, writable=True) as r:
        p = Patcher(r, backup=False)
        for off in (0, 4, r.data_start - 1):
            with pytest.raises(PatchAborted):
                p.poke_bytes(off, b"\xff")


def test_backup_and_restore(model):
    import os
    with open_gguf(model, writable=True) as r:
        Patcher(r).patch_scale(T, multiplier=2.0)
    assert os.path.exists(model + ".bak")
    restore_backup(model)
    with open_gguf(model) as r:
        pass  # still a valid file after restore


def test_scale_map(model):
    with open_gguf(model, writable=True) as r:
        n = r.tensors[T].blocks_count
        Patcher(r, backup=False).patch_scale(T, scale_map=np.full(n, 0.5, np.float32))
        np.testing.assert_allclose(r.scales(T), 0.5, rtol=1e-3)
        with pytest.raises(PatchError):
            Patcher(r, backup=False).patch_scale(T, scale_map=np.ones(3, np.float32))


# ---- hex editor ---------------------------------------------------------------------
def test_locate_and_block_view(model):
    with open_gguf(model) as r:
        off = tensor_start(r, "blk.0.attn_q.weight", 3)
        assert locate(r, off)["field"] == "d" and locate(r, off)["block"] == 3
        assert locate(r, off + 2)["field"] == "qs"
        assert locate(r, 0)["region"] == "header"
        assert locate(r, r.data_start - 1)["region"] in ("padding", "tensor_info")
        bv = block_view(r, "blk.0.attn_q.weight", 3)
        assert len(bv["dequant"]) == 32 and bv["fields"][0]["name"] == "d"
        v = hex_view(r, off, 64)
        assert v["rows"][0]["tags"][0].endswith("#3.d")
        with pytest.raises(IndexError):
            block_view(r, "blk.0.attn_q.weight", 10**9)


def test_poke_roundtrip(model):
    with open_gguf(model, writable=True) as r:
        p = Patcher(r, backup=False)
        off = tensor_start(r, "blk.0.attn_q.weight", 0)
        old = bytes(r.mmap[off:off + 2])
        p.poke_bytes(off, parse_hex("00 3c"))      # fp16 1.0
        assert r.scales("blk.0.attn_q.weight")[0] == 1.0
        p.revert()
        assert bytes(r.mmap[off:off + 2]) == old


def test_parse_hex():
    assert parse_hex("0xDE ad,be ef") == b"\xde\xad\xbe\xef"
    for bad in ("zz", "abc"):
        with pytest.raises(ValueError):
            parse_hex(bad)
