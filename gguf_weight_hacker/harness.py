"""Experiment harness: scripted patch -> evaluate -> revert loops with sweeps.

Programmatic use::

    from gguf_weight_hacker.harness import Harness
    with Harness("model.gguf", workdir="runs/exp1") as h:
        h.baseline()
        h.trial("blk.3.ffn_down.weight", multiplier=1.2)
        h.sweep("blk.*.ffn_down.weight", "multiplier", [0.8, 0.9, 1.1, 1.2], per_layer=True)
        h.save("results.jsonl")

The harness always works on a *copy* of the model inside ``workdir`` so your original is never
touched, and each trial is reverted through the patcher journal (verified by checksum).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field

import numpy as np

from .evaluator import EvalResult, check_thresholds, evaluate, load_suite
from .inspector import GGUFReader, open_gguf
from .log import get_logger, progress
from .patcher import PatchError, Patcher
from .tracer import NumpyLlama, Steering

log = get_logger("harness")


@dataclass
class Trial:
    id: int
    tensors: list
    op: str
    value: float | None
    steer: dict | None
    metrics: dict
    verdict: dict
    seconds: float
    error: str | None = None

    def to_dict(self):
        return self.__dict__.copy()


class Harness:
    def __init__(self, model_path: str, workdir: str = "runs/default", suite: dict | str | None = None,
                 backend: str = "numpy", thresholds: dict | None = None):
        os.makedirs(workdir, exist_ok=True)
        self.workdir = workdir
        self.src = model_path
        self.path = os.path.join(workdir, "work_" + os.path.basename(model_path))
        if not os.path.exists(self.path):
            shutil.copy2(model_path, self.path)
            log.info("working copy -> %s", self.path)
        self.reader: GGUFReader = open_gguf(self.path, writable=True)
        self.patcher = Patcher(self.reader, backup=False, journal=True)
        self.model = NumpyLlama(self.reader)
        self.suite = suite if isinstance(suite, dict) else load_suite(suite)
        self.thresholds = thresholds or {}
        self.base: EvalResult | None = None
        self.trials: list[Trial] = []
        self._hash0 = None

    # ---- lifecycle -----------------------------------------------------------
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
    def close(self): self.reader.close()

    # ---- baseline ------------------------------------------------------------
    def baseline(self) -> EvalResult:
        self.model.invalidate()
        self.base = evaluate(self.model, self.suite)
        self._hash0 = self._fingerprint()
        log.info("baseline ppl=%.3f over %d tokens", self.base.perplexity, self.base.n_tokens)
        return self.base

    def _fingerprint(self) -> str:
        import hashlib
        h = hashlib.sha256(); h.update(bytes(self.reader.mmap[self.reader.data_start:])); return h.hexdigest()

    # ---- single trial --------------------------------------------------------
    def trial(self, tensors: str | list[str], *, multiplier: float | None = None, set_to: float | None = None,
              field: str = "d", steer: Steering | None = None, label: str | None = None) -> Trial:
        """Apply an edit (and/or steering), evaluate vs baseline, then revert."""
        if self.base is None:
            self.baseline()
        names = self._resolve(tensors, field)
        t0, err, metrics, verdict = time.time(), None, {}, {}
        n_before = len(self.patcher.journal)
        try:
            for n in names:
                self.patcher.patch_scale(n, field=field, multiplier=multiplier, set_to=set_to)
            self.model.invalidate()
            m = _SteeredModel(self.model, steer) if steer else self.model
            res = evaluate(m, self.suite, self.base.logprobs)
            metrics = res.to_dict(); metrics.pop("per_prompt", None)
            verdict = check_thresholds(res, self.base.to_dict(), **self.thresholds)
        except PatchError as e:
            err = f"aborted: {e}"; verdict = {"pass": False, "reasons": [err]}
        finally:
            self.patcher.revert(len(self.patcher.journal) - n_before)
            self.model.invalidate()
        if self._hash0 and self._fingerprint() != self._hash0:
            raise RuntimeError("revert failed: working copy differs from baseline")
        tr = Trial(len(self.trials), names, "multiplier" if multiplier is not None else "set_to",
                   multiplier if multiplier is not None else set_to,
                   {"layer": steer.layer, "alpha": steer.alpha} if steer else None,
                   metrics, verdict, time.time() - t0, err)
        self.trials.append(tr)
        return tr

    def _resolve(self, tensors, field: str = "d") -> list[str]:
        """Expand names/globs. Glob matches silently skip tensors that have no ``field`` scale
        (e.g. F32 norm weights), so ``blk.*.ffn_*`` does what you mean."""
        pats = [tensors] if isinstance(tensors, str) else list(tensors)
        names = []
        for p in pats:
            if any(c in p for c in "*?["):
                hit = [t.name for t in self.reader.find(p) if any(f.name == field for f in t.layout.scale_fields)]
            else:
                hit = [p] if p in self.reader.tensors else []
            if not hit:
                raise KeyError(f"no tensor with a {field!r} scale matches {p!r}")
            names += hit
        return names

    # ---- sweeps --------------------------------------------------------------
    def sweep(self, pattern: str, param: str, values, *, per_layer: bool = False, field: str = "d") -> list[Trial]:
        """Sweep ``multiplier``/``set_to`` over ``values`` for ``pattern``.

        With ``per_layer`` each matching tensor is tested alone (sensitivity map); otherwise all
        matches are edited together.
        """
        if param not in ("multiplier", "set_to"):
            raise ValueError("param must be 'multiplier' or 'set_to'")
        res = self._resolve(pattern, field)
        groups = [[n] for n in res] if per_layer else [res]
        out = []
        total = len(groups) * len(values)
        bar = progress(total=total, desc="sweep", unit="trial")
        for g in groups:
            for v in values:
                out.append(self.trial(g, **{param: v, "field": field})); bar.update(1)
        bar.close()
        return out

    def steer_sweep(self, vector: np.ndarray, layers, alphas) -> list[Trial]:
        return [self._steer_only(vector, l, a) for l in layers for a in alphas]

    def _steer_only(self, vector, layer, alpha) -> Trial:
        if self.base is None:
            self.baseline()
        t0 = time.time()
        res = evaluate(_SteeredModel(self.model, Steering(vector, layer, alpha)), self.suite, self.base.logprobs)
        m = res.to_dict(); m.pop("per_prompt", None)
        tr = Trial(len(self.trials), [], "steer", alpha, {"layer": layer, "alpha": alpha}, m,
                   check_thresholds(res, self.base.to_dict(), **self.thresholds), time.time() - t0)
        self.trials.append(tr)
        return tr

    # ---- output --------------------------------------------------------------
    def table(self) -> list[dict]:
        rows = []
        for t in self.trials:
            rows.append({"id": t.id, "tensors": ",".join(t.tensors)[:60] or "-", "op": t.op, "value": t.value,
                         "steer": t.steer, **{k: t.metrics.get(k) for k in ("perplexity", "kl_mean", "top1_agree")},
                         "pass": t.verdict.get("pass"), "error": t.error})
        return rows

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(self.workdir, "results.jsonl")
        with open(path, "w") as fh:
            if self.base:
                fh.write(json.dumps({"baseline": self.base.to_dict(), "model": self.src}) + "\n")
            for t in self.trials:
                fh.write(json.dumps(t.to_dict(), default=float) + "\n")
        log.info("saved %d trials -> %s", len(self.trials), path)
        return path


class _SteeredModel:
    """Adapter so evaluate() sees a model with steering baked into ``logprobs``."""

    def __init__(self, model: NumpyLlama, steer: Steering):
        self.m, self.s = model, steer
    def encode(self, t): return self.m.encode(t)
    def logprobs(self, ids): return self.m.logprobs(ids, steer=self.s)


def load_results(path: str) -> tuple[dict | None, list[dict]]:
    base, rows = None, []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if "baseline" in d:
                base = d["baseline"]
            else:
                rows.append(d)
    return base, rows


def run_spec(spec: dict | str) -> Harness:
    """Run a declarative experiment spec (dict or path to JSON/YAML).

    ::

        model: model.gguf
        workdir: runs/ffn
        suite: prompts.json          # optional
        experiments:
          - {tensors: "blk.*.ffn_down.weight", param: multiplier, values: [0.9, 1.1], per_layer: true}
    """
    if isinstance(spec, str):
        with open(spec) as fh:
            txt = fh.read()
        if spec.endswith((".yml", ".yaml")):
            import yaml
            spec = yaml.safe_load(txt)
        else:
            spec = json.loads(txt)
    h = Harness(spec["model"], spec.get("workdir", "runs/default"), spec.get("suite"),
                thresholds=spec.get("thresholds"))
    h.baseline()
    for e in spec.get("experiments", []):
        h.sweep(e["tensors"], e.get("param", "multiplier"), e["values"], per_layer=e.get("per_layer", False),
                field=e.get("field", "d"))
    h.save()
    return h
