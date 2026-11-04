"""gguf-weight-hacker: zero-copy inspection, scale/byte patching, steering and evaluation
for GGUF-format quantized language models.

Quick start::

    from gguf_weight_hacker import open_gguf
    with open_gguf("model.gguf") as r:
        print(r.summary())
        scales = r.scales("blk.0.ffn_down.weight")
"""
from __future__ import annotations

__version__ = "0.1.0"

from .constants import GGMLType, GGUFValueType  # noqa: F401
from .inspector import GGUFReader, TensorInfo, open_gguf, GGUFFormatError  # noqa: F401
from .quant_codec import (dequantize_tensor, dequantize_raw, extract_scales,  # noqa: F401
                          block_stats, native_types)

__all__ = [
    "__version__", "GGMLType", "GGUFValueType", "GGUFReader", "TensorInfo",
    "open_gguf", "GGUFFormatError", "dequantize_tensor", "dequantize_raw",
    "extract_scales", "block_stats", "native_types",
]
