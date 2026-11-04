"""Evaluation & verification: perplexity, KL drift vs baseline, weight-space drift.

A *suite* is a JSON file ``{"prompts": ["text", ...], "max_tokens": 128}``. A *baseline* stores the
per-prompt log-probabilities (``.npz``) and summary metrics (``.json``) of the unmodified model so a
modified model can be compared by perplexity delta and mean token-level KL divergence.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .log import get_logger, progress
from .quant_codec import dequantize_tensor

log = get_logger("evaluator")

DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, humans first landed on the Moon.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Water boils at one hundred degrees Celsius at sea level.",
]


def load_suite(path: str | None) -> dict:
    if path is None:
        return {"prompts": list(DEFAULT_PROMPTS), "max_tokens": 128}
    with open(path) as fh:
        s = json.load(fh)
    s.setdefault("max_tokens", 128)
    return s


def _nll(lp: np.ndarray, ids) -> np.ndarray:
    """Per-position negative log-likelihood of the next token."""
    return -lp[np.arange(len(ids) - 1), np.asarray(ids[1:])]


@dataclass
class EvalResult:
    perplexity: float
    mean_nll: float
    n_tokens: int
    kl_mean: float | None = None
    kl_max: float | None = None
    top1_agree: float | None = None
    per_prompt: list = field(default_factory=list)
    logprobs: list = field(default_factory=list, repr=False)

    def to_dict(self):
        return {k: getattr(self, k) for k in ("perplexity", "mean_nll", "n_tokens", "kl_mean", "kl_max",
                                              "top1_agree", "per_prompt")}


def evaluate(model, suite: dict | None = None, baseline_logprobs: list | None = None) -> EvalResult:
    """Run the suite on ``model`` (anything with ``encode`` and ``logprobs``)."""
    suite = suite or load_suite(None)
    nlls, per, lps, kls, agree = [], [], [], [], []
    for i, text in enumerate(progress(suite["prompts"], desc="eval", unit="prompt")):
        ids = model.encode(text)[: suite.get("max_tokens", 128)]
        if len(ids) < 2:
            continue
        lp = model.logprobs(ids)
        lps.append(lp)
        n = _nll(lp, ids)
        nlls.append(n)
        row = {"prompt": text[:60], "tokens": len(ids), "ppl": float(np.exp(n.mean()))}
        if baseline_logprobs is not None and i < len(baseline_logprobs):
            b = baseline_logprobs[i][:-1]
            # KL(base || new) per position
            kl = (np.exp(b) * (b - lp[:-1])).sum(-1)
            kls.append(kl)
            agree.append((b.argmax(-1) == lp[:-1].argmax(-1)).astype(np.float32))
            row["kl"] = float(kl.mean())
        per.append(row)
    allnll = np.concatenate(nlls)
    res = EvalResult(float(np.exp(allnll.mean())), float(allnll.mean()), int(allnll.size),
                     per_prompt=per, logprobs=lps)
    if kls:
        k = np.concatenate(kls)
        res.kl_mean, res.kl_max = float(k.mean()), float(k.max())
        res.top1_agree = float(np.concatenate(agree).mean())
    return res


def save_baseline(res: EvalResult, path: str) -> str:
    npz = os.path.splitext(path)[0] + ".npz"
    np.savez_compressed(npz, **{f"p{i}": lp.astype(np.float16) for i, lp in enumerate(res.logprobs)})
    with open(path, "w") as fh:
        json.dump({**res.to_dict(), "logprobs_file": os.path.basename(npz)}, fh, indent=2)
    log.info("baseline saved -> %s", path)
    return path


def load_baseline(path: str) -> tuple[dict, list]:
    with open(path) as fh:
        meta = json.load(fh)
    z = np.load(os.path.join(os.path.dirname(path), meta["logprobs_file"]))
    lps = [z[f"p{i}"].astype(np.float32) for i in range(len(z.files))]
    return meta, lps


def check_thresholds(res: EvalResult, baseline_meta: dict | None, *, max_ppl_ratio: float = 1.5,
                     max_kl: float = 0.5) -> dict:
    """Pass/fail verdict against simple regression thresholds."""
    out = {"pass": True, "reasons": []}
    if baseline_meta:
        ratio = res.perplexity / baseline_meta["perplexity"]
        out["ppl_ratio"] = ratio
        if ratio > max_ppl_ratio:
            out["pass"] = False; out["reasons"].append(f"perplexity x{ratio:.2f} > {max_ppl_ratio}")
    if res.kl_mean is not None and res.kl_mean > max_kl:
        out["pass"] = False; out["reasons"].append(f"KL {res.kl_mean:.3f} > {max_kl}")
    if not np.isfinite(res.perplexity):
        out["pass"] = False; out["reasons"].append("non-finite perplexity")
    return out


# --------------------------------------------------------------------------- weight-space
def weight_drift(reader_a, reader_b, tensors: list[str] | None = None) -> list[dict]:
    """Compare dequantized weights between two files (e.g. original vs patched)."""
    names = tensors or [n for n in reader_a.tensors if n in reader_b.tensors
                        and reader_a.tensors[n].size_bytes == reader_b.tensors[n].size_bytes]
    rows = []
    for n in progress(names, desc="drift", unit="tensor"):
        a = dequantize_tensor(reader_a, n).ravel(); b = dequantize_tensor(reader_b, n).ravel()
        d = b - a
        na, nd = float(np.linalg.norm(a)), float(np.linalg.norm(d))
        if nd == 0:
            continue
        cos = float(a @ b / (na * np.linalg.norm(b) + 1e-30))
        rows.append({"tensor": n, "rel_l2": nd / (na + 1e-30), "cosine": cos,
                     "max_abs": float(np.abs(d).max()), "changed_frac": float((d != 0).mean())})
    return rows
