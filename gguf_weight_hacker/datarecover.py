"""Approximate fine-tuning data from a pair of GGUF checkpoints (simplified SELECT).

Based on *Approximating Language Model Training Data from Weights* (Morris et al., 2025,
arXiv:2506.15553). Given a base model ``θ0`` and a fine-tuned model ``θf`` you supply a public
*seed corpus*; this module ranks the seed texts by how well their training gradient explains the
weight change ``Δ = θf − θ0`` and returns the subset that best reconstructs it. It does **not**
extract the real training data — it picks the closest proxy available in the corpus you provide.

Simplifications vs. the paper (all documented in :func:`recover`):

* only the output head (last layer) is differentiated, as in the paper. Its per-example gradient is
  computed in closed form: ``∇W ℓ = Σ_t (p_t − e_{y_t}) ⊗ h_t`` — no autograd needed;
* gradients are compressed with a Kronecker Johnson–Lindenstrauss projection
  ``R1ᵀ (∇W) R2`` (``d = d1·d2``, default 64·64 = 4096 as in the paper), so the full
  ``vocab × n_embd`` gradient is never materialised;
* synthetic intermediate checkpoints are linear interpolations ``θ̂_j = θ0 + (j/P)(θf − θ0)``;
* labels are pseudo-labels from the fine-tuned model (``argmax θf``) by default, or the text's own
  next tokens (``labels="text"``);
* selection maximises cos(Σ_{x∈B} g_x, Δ) greedily, which favours a *diverse* batch whose summed
  update points along Δ, rather than K near-duplicates.

The update direction assumes SGD-like training (``Δ ∝ −Σ∇ℓ``). Adam-trained fine-tunes still work in
practice (the paper reports this), but the alignment is weaker.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .inspector import GGUFReader, open_gguf
from .log import get_logger, progress
from .tracer import NumpyLlama

log = get_logger("datarecover")


# --------------------------------------------------------------------------- corpus
def load_corpus(path: str, limit: int | None = None) -> list[str]:
    """Read seed texts from ``.txt`` (one per line), ``.jsonl`` (``{"text": ...}``) or ``.json`` (list)."""
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            texts = [json.loads(l)["text"] for l in fh if l.strip()]
        elif path.endswith(".json"):
            d = json.load(fh)
            texts = [x if isinstance(x, str) else x["text"] for x in d]
        else:
            texts = [l.rstrip("\n") for l in fh if l.strip()]
    return texts[:limit] if limit else texts


# --------------------------------------------------------------------------- checkpoints
class InterpolatedLlama(NumpyLlama):
    """``(1 − t)·θ0 + t·θf`` evaluated lazily, tensor by tensor, from two GGUF files."""

    def __init__(self, base: NumpyLlama, ft: NumpyLlama, t: float):
        self.__dict__.update({k: v for k, v in base.__dict__.items() if k != "_cache"})
        self._base, self._ft, self.t, self._cache = base, ft, float(t), {}

    def W(self, name: str) -> np.ndarray:
        if name not in self._cache:
            a, b = self._base.W(name), self._ft.W(name)
            self._cache[name] = a if self.t == 0 else b if self.t == 1 else ((1 - self.t) * a + self.t * b).astype(np.float32)
        return self._cache[name]


def check_pair(base: GGUFReader, ft: GGUFReader) -> list[str]:
    """Return tensors whose weights differ; raise if the files are not the same architecture."""
    if set(base.tensors) != set(ft.tensors):
        missing = sorted(set(base.tensors) ^ set(ft.tensors))[:5]
        raise ValueError(f"base and fine-tuned models have different tensors (e.g. {missing})")
    changed = []
    for n, t in base.tensors.items():
        u = ft.tensors[n]
        if t.shape != u.shape:
            raise ValueError(f"{n}: shape {t.shape} vs {u.shape}")
        if t.type != u.type or base.quick_checksum(n) != ft.quick_checksum(n):
            changed.append(n)
    return changed


# --------------------------------------------------------------------------- gradients
class _Projector:
    """Kronecker JL projection for (V × E) matrices: M ↦ R1ᵀ M R2, flattened to d1·d2."""

    def __init__(self, V: int, E: int, d: int, seed: int):
        rng = np.random.default_rng(seed)
        side = max(1, int(round(np.sqrt(d))))
        d1, d2 = min(V, side), min(E, max(1, d // max(1, min(V, side))))
        self.R1 = np.eye(V, dtype=np.float32) if d1 >= V else (rng.standard_normal((V, d1)) / np.sqrt(d1)).astype(np.float32)
        self.R2 = np.eye(E, dtype=np.float32) if d2 >= E else (rng.standard_normal((E, d2)) / np.sqrt(d2)).astype(np.float32)
        self.dim = self.R1.shape[1] * self.R2.shape[1]

    def outer(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Project Σ_t A_t ⊗ H_t without forming it: (A R1)ᵀ (H R2)."""
        return ((A @ self.R1).T @ (H @ self.R2)).ravel()

    def matrix(self, M: np.ndarray) -> np.ndarray:
        return (self.R1.T @ M @ self.R2).ravel()


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


# --------------------------------------------------------------------------- selection
@dataclass
class RecoveryResult:
    method: str
    indices: list[int]
    scores: list[float]            # per-selected-example alignment with Δ
    gains: list[float]             # batch cosine after adding each example (select only)
    all_scores: list[float]        # per-example alignment for every seed text
    texts: list[str] = field(repr=False, default_factory=list)
    info: dict = field(default_factory=dict)

    def selected(self) -> list[dict]:
        return [{"rank": r, "index": i, "score": s, "batch_cos": g, "text": self.texts[i]}
                for r, (i, s, g) in enumerate(zip(self.indices, self.scores, self.gains + [None] * len(self.indices)))]

    def to_dict(self, with_texts: bool = True) -> dict:
        d = {"method": self.method, "info": self.info, "n_seed": len(self.all_scores), "k": len(self.indices)}
        d["selected"] = self.selected() if with_texts else [{"index": i, "score": s} for i, s in zip(self.indices, self.scores)]
        return d

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            for row in self.selected():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("wrote %d selected examples -> %s", len(self.indices), path)
        return path


def _greedy_batch(G: np.ndarray, delta: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    """Greedily grow B to maximise cos(Σ_{x∈B} g_x, Δ) (Nemhauser-style greedy)."""
    dn = delta / (np.linalg.norm(delta) + 1e-30)
    S = np.zeros_like(delta); chosen: list[int] = []; gains: list[float] = []
    avail = np.ones(len(G), bool)
    proj = G @ dn                                   # (N,) component along Δ
    gnorm2 = (G * G).sum(1)
    for _ in range(min(k, len(G))):
        # cos(S + g, Δ) for every candidate, vectorised
        num = (S @ dn) + proj
        den = np.sqrt(np.maximum((S * S).sum() + 2 * (G @ S) + gnorm2, 1e-30))
        c = np.where(avail, num / den, -np.inf)
        i = int(np.argmax(c))
        if not np.isfinite(c[i]):
            break
        chosen.append(i); gains.append(float(c[i])); avail[i] = False; S = S + G[i]
    return chosen, gains


def recover(base_path: str, ft_path: str, corpus: Sequence[str] | str, k: int = 100, *,
            method: str = "select", n_checkpoints: int = 3, proj_dim: int = 4096,
            max_tokens: int = 64, labels: str = "pseudo", seed: int = 0) -> RecoveryResult:
    """Rank ``corpus`` texts by how well they explain ``θf − θ0`` and return the top ``k``.

    method: ``select`` (paper, greedy batch) · ``topk`` (per-example cosine) ·
    ``loss_drop`` (NLL under θ0 minus NLL under θf) · ``random`` (baseline).
    """
    texts = load_corpus(corpus) if isinstance(corpus, str) else list(corpus)
    if not texts:
        raise ValueError("seed corpus is empty")
    if method not in ("select", "topk", "loss_drop", "random"):
        raise ValueError(f"unknown method {method!r}")
    rb, rf = open_gguf(base_path), open_gguf(ft_path)
    try:
        changed = check_pair(rb, rf)
        if not changed:
            raise ValueError("the two models have identical weights; nothing to recover")
        mb, mf = NumpyLlama(rb), NumpyLlama(rf)
        head = mb.head_name
        info = {"changed_tensors": len(changed), "head": head, "head_changed": head in changed,
                "n_checkpoints": n_checkpoints, "labels": labels, "max_tokens": max_tokens}
        if head not in changed:
            log.warning("output head %s is unchanged; last-layer gradients carry no signal", head)
        ids_all = [mb.encode(t)[:max_tokens] for t in texts]
        rng = np.random.default_rng(seed)

        if method == "random":
            idx = rng.permutation(len(texts))[:k].tolist()
            return RecoveryResult(method, idx, [0.0] * len(idx), [], [0.0] * len(texts), texts, info)

        if method == "loss_drop":
            sc = np.full(len(texts), -np.inf)
            for i, ids in enumerate(progress(ids_all, desc="loss-drop", unit="text")):
                if len(ids) < 2: continue
                tgt = np.asarray(ids[1:])
                nb = -mb.logprobs(ids)[np.arange(len(tgt)), tgt].mean()
                nf = -mf.logprobs(ids)[np.arange(len(tgt)), tgt].mean()
                sc[i] = nb - nf
            idx = np.argsort(-sc)[:k].tolist()
            return RecoveryResult(method, idx, sc[idx].tolist(), [], sc.tolist(), texts, info)

        # ---- gradient-alignment methods (select / topk) ----
        Wb, Wf = mb.W(head), mf.W(head)
        V, E = Wb.shape
        P = _Projector(V, E, proj_dim, seed)
        info["proj_dim"] = P.dim
        ts = [j / n_checkpoints for j in range(n_checkpoints)]          # θ̂_j for j = 0..P-1 (θ0 … just before θf)
        ckpts = [InterpolatedLlama(mb, mf, t) for t in ts]
        deltas = [P.matrix((Wf - ck.W(head))) for ck in ckpts]           # θf − θ̂_j, projected
        G = np.zeros((len(texts), P.dim * len(ckpts)), np.float32)
        for i, ids in enumerate(progress(ids_all, desc="per-example grads", unit="text")):
            if len(ids) < 2:
                continue
            if labels == "pseudo":
                y = mf.forward(ids)[:-1].argmax(-1)                      # θf's prediction at each position
            else:
                y = np.asarray(ids[1:])
            for j, ck in enumerate(ckpts):
                h = ck.forward(ids, return_hidden=True)[:-1]             # (T-1, E)
                p = _softmax(h @ ck.W(head).T)                            # (T-1, V)
                p[np.arange(len(y)), y] -= 1.0                            # ∂ℓ/∂logits
                g = -P.outer(p, h) / len(y)                               # descent direction −∇Wℓ
                G[i, j * P.dim:(j + 1) * P.dim] = g
        D = np.concatenate(deltas).astype(np.float32)
        dn = D / (np.linalg.norm(D) + 1e-30)
        per = (G @ dn) / (np.linalg.norm(G, axis=1) + 1e-30)            # cos(g_x, Δ)
        if method == "topk":
            idx = np.argsort(-per)[:k].tolist()
            return RecoveryResult(method, idx, per[idx].tolist(), [], per.tolist(), texts, info)
        idx, gains = _greedy_batch(G, D, k)
        info["final_batch_cos"] = gains[-1] if gains else None
        return RecoveryResult(method, idx, per[idx].tolist(), gains, per.tolist(), texts, info)
    finally:
        rb.close(); rf.close()


# --------------------------------------------------------------------------- evaluation
def vocab_containment(selected: Sequence[str], reference: Sequence[str]) -> float:
    """Paper's lexical metric: max(|A∩B|/|A|, |A∩B|/|B|) over word sets."""
    a = {w for t in selected for w in t.lower().split()}
    b = {w for t in reference for w in t.lower().split()}
    if not a or not b:
        return 0.0
    i = len(a & b)
    return max(i / len(a), i / len(b))


def compare_methods(base_path, ft_path, corpus, k=50, reference: Sequence[str] | None = None,
                    labels_of: Sequence | None = None, **kw) -> list[dict]:
    """Run every method; report vocab containment vs ``reference`` and/or label purity vs ``labels_of``."""
    texts = load_corpus(corpus) if isinstance(corpus, str) else list(corpus)
    rows = []
    for m in ("random", "loss_drop", "topk", "select"):
        r = recover(base_path, ft_path, texts, k, method=m, **kw)
        row = {"method": m, "k": len(r.indices)}
        if reference is not None:
            row["vocab_containment"] = vocab_containment([texts[i] for i in r.indices], reference)
        if labels_of is not None:
            from collections import Counter
            c = Counter(labels_of[i] for i in r.indices)
            row["label_mix"] = dict(c)
        rows.append(row)
    return rows
