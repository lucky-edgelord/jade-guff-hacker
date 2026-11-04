"""Programmatic API tour: inspect -> patch -> evaluate -> sweep -> visualize.

    python examples/quickstart.py            # builds a tiny synthetic model and runs everything
    python examples/quickstart.py my.gguf    # run on your own (small) model
"""
import sys, tempfile, os
import numpy as np
import gguf_weight_hacker as gh
from gguf_weight_hacker import viz

work = tempfile.mkdtemp()
path = sys.argv[1] if len(sys.argv) > 1 else gh.build_tiny_model(os.path.join(work, "tiny.gguf"))

# 1. inspect (zero-copy, mmap) -------------------------------------------------------------
with gh.open_gguf(path) as r:
    print(r.summary()["architecture"], r.summary()["type_counts"])
    name = next(t.name for t in r if t.layout.scale_fields and "ffn_down" in t.name)
    print(name, r[name].to_dict()["offset_bytes"], gh.block_stats(r, name)["d"])
    viz.byte_map(r, os.path.join(work, "bytemap.png"))

# 2. harness: patch -> evaluate -> revert, always on a working copy ------------------------
with gh.Harness(path, workdir=os.path.join(work, "run")) as h:
    h.baseline()
    h.sweep("blk.*.ffn_down.weight", "multiplier", [0.5, 1.5], per_layer=True)     # sensitivity map
    vec = gh.contrast_vector(h.model, ["happy"], ["sad"], layer=1)                  # steering vector
    h.steer_sweep(vec, layers=[1, 2], alphas=[1.0, 4.0])
    for row in h.table()[:4]: print(row)
    h.save()
    viz.sensitivity_heatmap(h.table(), out=os.path.join(work, "sensitivity.png"))

# 3. edit a file in place (with backup + journal + validation) -----------------------------
with gh.open_gguf(path, writable=True) as r:
    p = gh.Patcher(r)
    print(p.patch_scale(name, multiplier=1.1, dry_run=True).stats)    # preview
    # p.patch_scale(name, multiplier=1.1); p.revert()                 # apply / undo
print("artifacts in", work)
