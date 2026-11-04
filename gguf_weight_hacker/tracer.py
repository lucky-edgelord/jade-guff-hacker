"""Activation tracing and steering.

Two backends share one interface:

* :class:`NumpyLlama` — a dependency-free reference forward pass for llama-style GGUF models
  (RMSNorm, RoPE, multi-head attention with GQA, SwiGLU). It dequantizes weights straight out of
  the memory map, so edits made by the patcher are visible after :meth:`NumpyLlama.invalidate`.
  It supports residual-stream capture and steering-vector injection at any layer. It is meant for
  small models, tests and quick experiments, not for throughput.
* :class:`LlamaCppBackend` — optional bridge to ``llama-cpp-python`` for perplexity-style scoring
  on full-size models. Steering at runtime for llama.cpp is done by exporting a control-vector
  GGUF with :func:`export_control_vector` and passing it to llama.cpp (``--control-vector``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .inspector import GGUFReader
from .log import get_logger
from .quant_codec import dequantize_tensor

log = get_logger("tracer")


# --------------------------------------------------------------------------- tokenizer
class SimpleTokenizer:
    """Greedy longest-match tokenizer over the GGUF vocab; byte fallback when none exists."""

    def __init__(self, reader: GGUFReader, vocab_size: int):
        self.vocab_size = vocab_size
        kv = reader.kv.get("tokenizer.ggml.tokens")
        self.tokens = [str(t) for t in kv.value] if kv is not None else None
        if self.tokens:
            self.lookup = {t.replace("▁", " "): i for i, t in enumerate(self.tokens)}
            self.maxlen = max(len(t) for t in self.lookup) if self.lookup else 1
        bos = reader.kv.get("tokenizer.ggml.bos_token_id")
        self.bos = int(bos.value) if bos is not None else None

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids: list[int] = [self.bos] if (add_bos and self.bos is not None) else []
        if not self.tokens:
            return ids + [b % self.vocab_size for b in text.encode("utf-8")]
        i, text = 0, " " + text
        while i < len(text):
            for L in range(min(self.maxlen, len(text) - i), 0, -1):
                tid = self.lookup.get(text[i:i + L])
                if tid is not None:
                    ids.append(tid); i += L; break
            else:
                ids.append(ord(text[i]) % self.vocab_size); i += 1
        return ids


# --------------------------------------------------------------------------- steering
@dataclass
class Steering:
    """Add ``alpha * vector`` to the residual stream at the output of ``layer``."""
    vector: np.ndarray
    layer: int
    alpha: float = 1.0
    positions: str = "all"        # "all" | "last"

    def apply(self, h: np.ndarray) -> np.ndarray:
        v = (self.alpha * self.vector).astype(h.dtype)
        if self.positions == "last":
            h = h.copy(); h[-1] += v; return h
        return h + v


def load_vector(path: str) -> np.ndarray:
    v = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",")
    return np.asarray(v, np.float32).reshape(-1)


# --------------------------------------------------------------------------- numpy backend
def _rmsnorm(x, w, eps):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _rope(x, pos, theta):
    """Interleaved-pair RoPE. x: (T, H, D)."""
    D = x.shape[-1]
    inv = theta ** (-np.arange(0, D, 2, dtype=np.float32) / D)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class NumpyLlama:
    """Reference llama forward pass reading weights from a :class:`GGUFReader`."""

    def __init__(self, reader: GGUFReader):
        self.r = reader
        arch = reader.kv["general.architecture"].value
        g = lambda k, d=None: (reader.kv[f"{arch}.{k}"].value if f"{arch}.{k}" in reader.kv else d)
        self.n_layers = int(g("block_count"))
        self.n_embd = int(g("embedding_length"))
        self.n_head = int(g("attention.head_count"))
        self.n_kv = int(g("attention.head_count_kv", self.n_head))
        self.eps = float(g("attention.layer_norm_rms_epsilon", 1e-5))
        self.theta = float(g("rope.freq_base", 10000.0))
        self.head_dim = self.n_embd // self.n_head
        self.vocab = reader.tensors["token_embd.weight"].shape[1]
        self.tok = SimpleTokenizer(reader, self.vocab)
        self._cache: dict[str, np.ndarray] = {}

    # weights --------------------------------------------------------------------
    def invalidate(self):
        """Drop cached dequantized weights (call after patching the file)."""
        self._cache.clear()

    def W(self, name: str) -> np.ndarray:
        if name not in self._cache:
            ti = self.r.tensors[name]
            if len(ti.shape) == 1:
                self._cache[name] = dequantize_tensor(self.r, name).astype(np.float32)
            else:
                self._cache[name] = dequantize_tensor(self.r, name).astype(np.float32)
        return self._cache[name]

    # forward --------------------------------------------------------------------
    def forward(self, ids: Sequence[int], steer: Steering | Sequence[Steering] | None = None,
                capture: bool = False):
        """Return logits (T, vocab); with ``capture`` also ``{layer: residual (T, n_embd)}``."""
        steers = [steer] if isinstance(steer, Steering) else list(steer or [])
        T = len(ids)
        h = self.W("token_embd.weight")[np.asarray(ids)]
        pos = np.arange(T, dtype=np.float32)
        mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
        captured: dict[int, np.ndarray] = {}
        rep = self.n_head // self.n_kv
        for l in range(self.n_layers):
            p = f"blk.{l}."
            x = _rmsnorm(h, self.W(p + "attn_norm.weight"), self.eps)
            q = (x @ self.W(p + "attn_q.weight").T).reshape(T, self.n_head, self.head_dim)
            k = (x @ self.W(p + "attn_k.weight").T).reshape(T, self.n_kv, self.head_dim)
            v = (x @ self.W(p + "attn_v.weight").T).reshape(T, self.n_kv, self.head_dim)
            q, k = _rope(q, pos, self.theta), _rope(k, pos, self.theta)
            k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
            att = np.einsum("thd,shd->hts", q, k) / np.sqrt(self.head_dim) + mask
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            o = np.einsum("hts,shd->thd", att, v).reshape(T, self.n_embd)
            h = h + o @ self.W(p + "attn_output.weight").T
            x = _rmsnorm(h, self.W(p + "ffn_norm.weight"), self.eps)
            ff = _silu(x @ self.W(p + "ffn_gate.weight").T) * (x @ self.W(p + "ffn_up.weight").T)
            h = h + ff @ self.W(p + "ffn_down.weight").T
            for s in steers:
                if s.layer == l:
                    h = s.apply(h)
            if capture:
                captured[l] = h.copy()
        h = _rmsnorm(h, self.W("output_norm.weight"), self.eps)
        head = "output.weight" if "output.weight" in self.r.tensors else "token_embd.weight"
        logits = h @ self.W(head).T
        return (logits, captured) if capture else logits

    # helpers --------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def logprobs(self, ids, steer=None) -> np.ndarray:
        lg = self.forward(ids, steer=steer)
        lg = lg - lg.max(-1, keepdims=True)
        return lg - np.log(np.exp(lg).sum(-1, keepdims=True))

    def generate(self, prompt: str, n: int = 16, steer=None, greedy: bool = True, seed: int = 0) -> list[int]:
        rng = np.random.default_rng(seed)
        ids = self.encode(prompt)
        for _ in range(n):
            lp = self.logprobs(ids, steer=steer)[-1]
            ids.append(int(lp.argmax()) if greedy else int(rng.choice(len(lp), p=np.exp(lp))))
        return ids

    def activation_norms(self, text: str) -> dict[int, float]:
        _, cap = self.forward(self.encode(text), capture=True)
        return {l: float(np.linalg.norm(a, axis=-1).mean()) for l, a in cap.items()}


# --------------------------------------------------------------------------- vectors
def contrast_vector(model: NumpyLlama, positive: Sequence[str], negative: Sequence[str],
                    layer: int, normalize: bool = True) -> np.ndarray:
    """Difference-of-means steering vector from contrastive prompt sets (last-token residuals)."""
    def mean_last(texts):
        acc = []
        for t in texts:
            _, cap = model.forward(model.encode(t), capture=True)
            acc.append(cap[layer][-1])
        return np.mean(acc, axis=0)
    v = (mean_last(positive) - mean_last(negative)).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if (normalize and n > 0) else v


def export_control_vector(path: str, vectors: dict[int, np.ndarray], model_hint: str = "llama") -> str:
    """Write a llama.cpp-compatible control-vector GGUF (``--control-vector path``).

    ``vectors`` maps layer index -> direction (n_embd,). Layers are 0-based here; llama.cpp stores
    ``direction.{il+1}``... please see llama.cpp's ``control-vector`` docs for layer ranges.
    """
    import gguf
    n_layers = max(vectors) + 1
    w = gguf.GGUFWriter(path, "controlvector")
    w.add_string("controlvector.model_hint", model_hint)
    w.add_uint32("controlvector.layer_count", n_layers)
    for l, v in sorted(vectors.items()):
        w.add_tensor(f"direction.{l + 1}", np.asarray(v, np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    log.info("control vector written -> %s (%d layers)", path, len(vectors))
    return path


# --------------------------------------------------------------------------- llama.cpp
class LlamaCppBackend:
    """Optional scoring backend using ``llama-cpp-python`` (experimental, untested here)."""

    def __init__(self, path: str, n_ctx: int = 512, **kw):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("pip install llama-cpp-python to use the llama.cpp backend") from e
        self.llm = Llama(model_path=path, n_ctx=n_ctx, logits_all=True, verbose=False, **kw)

    def logprobs(self, ids, steer=None) -> np.ndarray:
        if steer is not None:
            raise NotImplementedError("runtime steering via llama.cpp: use export_control_vector()")
        self.llm.reset(); self.llm.eval(list(ids))
        lg = np.asarray(self.llm.scores[: len(ids)], np.float32)
        lg = lg - lg.max(-1, keepdims=True)
        return lg - np.log(np.exp(lg).sum(-1, keepdims=True))

    def encode(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"))

    def invalidate(self):  # file edits require a reload
        raise NotImplementedError("reload the model after editing the file")
