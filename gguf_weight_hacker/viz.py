"""Visualizers (matplotlib, headless) and a self-contained HTML report.

Every plot function returns PNG ``bytes`` and optionally writes to ``out``. The web dashboard
renders its own interactive views from JSON; these are for scripts, reports and the ``/api/plot``
endpoint.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os

import numpy as np

from .inspector import GGUFReader
from .quant_codec import dequantize_tensor

_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D", "#BAB0AC"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
    return plt


def _finish(fig, out):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    data = buf.getvalue()
    if out:
        with open(out, "wb") as fh:
            fh.write(data)
    return data


def byte_map(reader: GGUFReader, out: str | None = None, max_tensors: int = 400) -> bytes:
    """File layout: header/KV/info/padding then every tensor as a coloured span, by quant type."""
    plt = _plt()
    types = sorted({t.type.name for t in reader})
    col = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(types)}
    fig, ax = plt.subplots(figsize=(11, 2.6))
    fs = reader.filesize()
    ax.broken_barh([(0, reader.data_start)], (0, 1), facecolors="#888888")
    for t in list(reader)[:max_tensors]:
        ax.broken_barh([(t.offset_bytes, t.size_bytes)], (0, 1), facecolors=col[t.type.name], linewidth=0)
    ax.set_xlim(0, fs); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("file offset (bytes)")
    ax.set_title(f"File byte map — header/metadata (grey) + {len(reader.tensors)} tensors")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#888888", label="header+KV")] + [Patch(color=c, label=t) for t, c in col.items()],
              ncol=len(col) + 1, loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False)
    ax.grid(False)
    return _finish(fig, out)


def scale_histogram(reader: GGUFReader, tensor: str, field: str = "d", out: str | None = None) -> bytes:
    plt = _plt()
    s = reader.scales(tensor, field)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(s[np.isfinite(s)], bins=80, color=_PALETTE[0])
    ax.set_title(f"{tensor} — {field} (n={s.size})"); ax.set_xlabel(field); ax.set_ylabel("blocks")
    return _finish(fig, out)


def scale_heatmap(reader: GGUFReader, tensor: str, field: str = "d", out: str | None = None) -> bytes:
    """Block scales laid out as (rows x blocks-per-row) — outlier rows/columns jump out."""
    plt = _plt()
    ti = reader.tensors[tensor]
    s = reader.scales(tensor, field)
    bpr = max(1, ti.shape[0] // ti.layout.block_elems)
    grid = s[: (s.size // bpr) * bpr].reshape(-1, bpr)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(f"{tensor} — {field} heatmap"); ax.set_xlabel("block in row"); ax.set_ylabel("row"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    return _finish(fig, out)


def layer_stats(reader: GGUFReader, kind: str = "ffn_down", field: str = "d", out: str | None = None) -> bytes:
    """Mean/std of block scales per layer for one tensor kind (e.g. ``attn_v``)."""
    plt = _plt()
    xs, mu, sd = [], [], []
    for l in range(reader.num_layers()):
        n = f"blk.{l}.{kind}.weight"
        if n in reader.tensors and field in (reader.tensors[n].layout.field(field).name,):
            s = reader.scales(n, field); xs.append(l); mu.append(s.mean()); sd.append(s.std())
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.errorbar(xs, mu, yerr=sd, fmt="o-", color=_PALETTE[0], capsize=3)
    ax.set_title(f"{kind}.{field} per layer (mean ± std)"); ax.set_xlabel("layer")
    return _finish(fig, out)


def diff_plot(reader_a: GGUFReader, reader_b: GGUFReader, tensor: str, out: str | None = None) -> bytes:
    """Before/after weight comparison: scatter of dequantized values + residual histogram."""
    plt = _plt()
    a = dequantize_tensor(reader_a, tensor).ravel(); b = dequantize_tensor(reader_b, tensor).ravel()
    idx = np.random.default_rng(0).choice(a.size, size=min(a.size, 20000), replace=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.scatter(a[idx], b[idx], s=1, alpha=0.3, color=_PALETTE[0])
    lim = float(np.abs(a).max()); ax1.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax1.set_xlabel("original"); ax1.set_ylabel("modified"); ax1.set_title("weights")
    ax2.hist((b - a)[idx], bins=80, color=_PALETTE[1]); ax2.set_title("delta"); ax2.set_xlabel("modified − original")
    fig.suptitle(tensor, y=1.02)
    return _finish(fig, out)


def drift_bars(rows: list[dict], out: str | None = None) -> bytes:
    plt = _plt()
    rows = sorted(rows, key=lambda r: -r["rel_l2"])[:30]
    fig, ax = plt.subplots(figsize=(8, max(2, 0.28 * len(rows) + 1)))
    ax.barh([r["tensor"] for r in rows][::-1], [r["rel_l2"] for r in rows][::-1], color=_PALETTE[3])
    ax.set_xlabel("relative L2 change"); ax.set_title("Weight drift by tensor")
    return _finish(fig, out)


def activation_plot(norms: dict[int, float] | dict[str, dict], out: str | None = None) -> bytes:
    """Residual-stream norm by layer. Pass ``{layer: norm}`` or ``{label: {layer: norm}}`` to overlay runs."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    series = norms if norms and isinstance(next(iter(norms.values())), dict) else {"run": norms}
    for i, (lab, d) in enumerate(series.items()):
        ax.plot(list(d.keys()), list(d.values()), "o-", color=_PALETTE[i % 8], label=lab)
    ax.set_xlabel("layer"); ax.set_ylabel("mean ‖residual‖"); ax.set_title("Residual stream norm")
    if len(series) > 1:
        ax.legend(frameon=False)
    return _finish(fig, out)


def sweep_curves(rows: list[dict], metric: str = "perplexity", out: str | None = None) -> bytes:
    """Metric vs swept value, one line per tensor group (from ``Harness.table()`` rows)."""
    plt = _plt()
    groups: dict[str, list] = {}
    for r in rows:
        if r.get("value") is not None and r.get(metric) is not None and r.get("op") != "steer":
            groups.setdefault(r["tensors"], []).append((r["value"], r[metric]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (g, pts) in enumerate(groups.items()):
        pts.sort(); ax.plot(*zip(*pts), "o-", color=_PALETTE[i % 8], label=g[:28], alpha=0.85)
    ax.set_xlabel("value"); ax.set_ylabel(metric); ax.set_title(f"{metric} vs edit")
    if 1 < len(groups) <= 12:
        ax.legend(frameon=False, fontsize=7, ncol=2)
    return _finish(fig, out)


def sensitivity_heatmap(rows: list[dict], metric: str = "kl_mean", out: str | None = None) -> bytes:
    """Layer x value heatmap for per-layer sweeps — shows which layers are fragile."""
    plt = _plt()
    import re
    cells: dict = {}
    for r in rows:
        m = re.search(r"blk\.(\d+)\.", r["tensors"])
        if m and r.get(metric) is not None and r.get("value") is not None:
            cells[(int(m.group(1)), r["value"])] = r[metric]
    if not cells:
        raise ValueError("no per-layer rows to plot")
    layers = sorted({k[0] for k in cells}); vals = sorted({k[1] for k in cells})
    grid = np.full((len(layers), len(vals)), np.nan)
    for (l, v), x in cells.items():
        grid[layers.index(l), vals.index(v)] = x
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(vals), 1 + 0.4 * len(layers)))
    im = ax.imshow(grid, cmap="magma_r", aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(vals)), [f"{v:g}" for v in vals]); ax.set_yticks(range(len(layers)), layers)
    ax.set_xlabel("value"); ax.set_ylabel("layer"); ax.set_title(f"Layer sensitivity ({metric})")
    fig.colorbar(im, ax=ax, fraction=0.04)
    return _finish(fig, out)


# --------------------------------------------------------------------------- report
def html_report(reader: GGUFReader, out: str, *, results_rows: list[dict] | None = None,
                tensors: list[str] | None = None) -> str:
    """Write a single-file HTML report with embedded plots."""
    def img(b: bytes) -> str:
        return f'<img src="data:image/png;base64,{base64.b64encode(b).decode()}"/>'

    s = reader.summary()
    tensors = tensors or [t.name for t in reader if t.layout.scale_fields][:4]
    parts = [f"<h1>GGUF report</h1><pre>{html.escape(json.dumps({k: v for k, v in s.items() if k != 'kv_keys'}, indent=2))}</pre>",
             "<h2>Byte map</h2>" + img(byte_map(reader))]
    for kind in ("attn_v", "ffn_down"):
        try:
            parts.append(f"<h2>{kind} scales per layer</h2>" + img(layer_stats(reader, kind)))
        except Exception:
            pass
    for t in tensors:
        parts.append(f"<h2>{html.escape(t)}</h2>" + img(scale_histogram(reader, t)) + img(scale_heatmap(reader, t)))
    if results_rows:
        parts.append("<h2>Experiments</h2>" + img(sweep_curves(results_rows)))
        try:
            parts.append(img(sensitivity_heatmap(results_rows)))
        except ValueError:
            pass
    doc = ("<!doctype html><meta charset=utf-8><title>GGUF report</title><style>body{font:14px system-ui;"
           "max-width:980px;margin:2rem auto;padding:0 1rem}img{max-width:100%;margin:.5rem 0}pre{background:#f4f4f4;"
           "padding:1rem;overflow:auto}</style>" + "".join(parts))
    with open(out, "w") as fh:
        fh.write(doc)
    return out
