"""Build tiny synthetic GGUF models for tests, demos and the dashboard.

Uses the reference ``gguf.GGUFWriter`` so the files are byte-compatible with llama.cpp tooling.
Produces a miniature llama-shaped model with a handful of layers and mixed quant types.
"""
from __future__ import annotations

import os

import numpy as np

from .log import get_logger

log = get_logger("testmodel")


def build_tiny_model(path: str, *, n_layers: int = 4, n_embd: int = 256, n_ff: int = 512,
                     seed: int = 0, quant: str = "Q4_K") -> str:
    """Write a small valid GGUF llama model to ``path``; returns the path."""
    import gguf
    from gguf.constants import GGMLQuantizationType as T

    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(path, "llama")
    w.add_name("tiny-test")
    w.add_context_length(512)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layers)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(8)
    w.add_head_count_kv(8)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(7)  # llama ftype MOSTLY_Q8_0 (the tensors below are Q8_0)

    def q(name, arr, qtype):
        """Quantize via reference where supported, else store F16."""
        from gguf.quants import quantize
        try:
            data = quantize(arr.astype(np.float32), qtype)
            w.add_tensor(name, data, raw_dtype=qtype)
        except NotImplementedError:
            w.add_tensor(name, arr.astype(np.float16))

    # token embedding + output norm/head (F16 / Q)
    w.add_tensor("token_embd.weight", (rng.standard_normal((32, n_embd)) * 0.02).astype(np.float16))
    w.add_tensor("output_norm.weight", np.ones(n_embd, np.float32))
    for i in range(n_layers):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", (rng.standard_normal(n_embd) * 0.1 + 1).astype(np.float32))
        w.add_tensor(p + "ffn_norm.weight", (rng.standard_normal(n_embd) * 0.1 + 1).astype(np.float32))
        # Q8_0 (reference quantize supports it) for attn, Q4_K-or-F16 for ffn
        for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
            q(p + nm + ".weight", rng.standard_normal((n_embd, n_embd)) * 0.05, T.Q8_0)
        q(p + "ffn_gate.weight", rng.standard_normal((n_ff, n_embd)) * 0.05, T.Q8_0)
        q(p + "ffn_up.weight", rng.standard_normal((n_ff, n_embd)) * 0.05, T.Q8_0)
        q(p + "ffn_down.weight", rng.standard_normal((n_embd, n_ff)) * 0.05, T.Q8_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log.info("wrote tiny model -> %s (%d bytes)", path, os.path.getsize(path))
    return path


if __name__ == "__main__":
    import sys
    build_tiny_model(sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf")
