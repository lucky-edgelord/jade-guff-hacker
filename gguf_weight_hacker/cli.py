"""``gguf-hack`` command line interface."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .log import configure


def _out(obj, as_json=True):
    print(json.dumps(obj, indent=2, default=float) if as_json else obj)


def cmd_inspect(a):
    from .inspector import open_gguf
    with open_gguf(a.model) as r:
        if a.tensor:
            _out(r[a.tensor].to_dict()) if a.json else _print_tensor(r[a.tensor])
        elif a.json:
            s = r.summary(); s["tensors"] = [t.to_dict() for t in r]
            if a.kv: s["kv"] = {k: e.summary() for k, e in r.kv.items()}
            _out(s)
        else:
            s = r.summary()
            print(f'{s["path"]}\n  GGUF v{s["version"]}  arch={s["architecture"]}  ftype={s["file_type"]}  '
                  f'layers={s["n_layers"]}  tensors={s["n_tensors"]}  size={s["size_bytes"]:,} B')
            print("  types:", ", ".join(f"{k}x{v}" for k, v in s["type_counts"].items()))
            if a.kv:
                for k, e in r.kv.items(): print(f"  {k} = {e.summary()}")
            for t in r:
                print(f"  {t.offset_bytes:>12,}  {t.type.name:<7} {str(list(t.shape)):<18} {t.size_bytes:>11,}  {t.name}")


def _print_tensor(t):
    d = t.to_dict()
    for k, v in d.items(): print(f"{k:>18}: {v}")


def cmd_unpack(a):
    from .inspector import open_gguf
    with open_gguf(a.model) as r:
        s = r.scales(a.tensor, a.field)
    if a.out.endswith(".npy"): np.save(a.out, s)
    else: np.savetxt(a.out, s, delimiter=",")
    print(f"wrote {s.size} {a.field} values -> {a.out}  (mean={s.mean():.6g} std={s.std():.6g})")


def cmd_patch(a):
    from .inspector import open_gguf
    from .patcher import Patcher
    with open_gguf(a.model, writable=not a.dry_run) as r:
        p = Patcher(r, backup=not a.no_backup)
        kw = dict(field=a.field, dry_run=a.dry_run)
        if a.scale_map: kw["scale_map"] = np.load(a.scale_map)
        elif a.multiplier is not None: kw["multiplier"] = a.multiplier
        elif a.set_to is not None: kw["set_to"] = a.set_to
        else: sys.exit("give --scale-map, --multiplier or --set")
        for t in (r.find(a.tensor) or [r[a.tensor]]):
            _out(p.patch_scale(t.name, **kw).to_dict())
        if a.journal and not a.dry_run: p.save_journal(a.journal)


def cmd_restore(a):
    from .patcher import restore_backup
    print("restored", restore_backup(a.model))


def cmd_steer(a):
    from .inspector import open_gguf
    from .tracer import NumpyLlama, Steering, load_vector, export_control_vector
    if a.export:
        export_control_vector(a.export, {a.layer: load_vector(a.vector) * a.alpha}); return
    with open_gguf(a.model) as r:
        m = NumpyLlama(r)
        s = Steering(load_vector(a.vector), a.layer, a.alpha)
        ids = m.generate(a.prompt, a.tokens, steer=s)
        base = m.generate(a.prompt, a.tokens)
        print("baseline ids:", base[-a.tokens:]); print("steered  ids:", ids[-a.tokens:])


def cmd_eval(a):
    from .inspector import open_gguf
    from .tracer import NumpyLlama
    from .evaluator import evaluate, load_suite, save_baseline, load_baseline, check_thresholds
    with open_gguf(a.model) as r:
        m = NumpyLlama(r)
        meta, lps = load_baseline(a.baseline) if a.baseline and not a.save_baseline else (None, None)
        res = evaluate(m, load_suite(a.eval_dataset), lps)
        if a.save_baseline: save_baseline(res, a.save_baseline)
        out = res.to_dict()
        if meta: out["verdict"] = check_thresholds(res, meta, max_ppl_ratio=a.max_ppl_ratio, max_kl=a.max_kl)
        _out(out)
        if meta and not out["verdict"]["pass"]: sys.exit(2)


def cmd_hexdump(a):
    from .inspector import open_gguf
    from .hexedit import hexdump_text, tensor_start
    with open_gguf(a.model) as r:
        off = a.offset if a.offset is not None else (tensor_start(r, a.tensor, a.block) if a.tensor else 0)
        print(hexdump_text(r, off, a.length, color=sys.stdout.isatty()))


def cmd_poke(a):
    from .inspector import open_gguf
    from .hexedit import parse_hex, tensor_start
    from .patcher import Patcher
    with open_gguf(a.model, writable=not a.dry_run) as r:
        p = Patcher(r, backup=not a.no_backup)
        off = a.offset if a.offset is not None else tensor_start(r, a.tensor, a.block)
        _out(p.poke_bytes(off, parse_hex(a.hex), dry_run=a.dry_run).to_dict())


def cmd_viz(a):
    from .inspector import open_gguf
    from . import viz
    with open_gguf(a.model) as r:
        if a.kind == "report":
            print(viz.html_report(r, a.out, tensors=[a.tensor] if a.tensor else None)); return
        fn = {"bytemap": lambda: viz.byte_map(r, a.out), "hist": lambda: viz.scale_histogram(r, a.tensor, a.field, a.out),
              "heatmap": lambda: viz.scale_heatmap(r, a.tensor, a.field, a.out),
              "layers": lambda: viz.layer_stats(r, a.tensor or "ffn_down", a.field, a.out)}[a.kind]
        fn(); print("wrote", a.out)


def cmd_harness(a):
    from .harness import run_spec
    h = run_spec(a.spec)
    for row in h.table(): print(row)
    h.close()


def cmd_recover(a):
    from .datarecover import recover, load_corpus, vocab_containment
    r = recover(a.base, a.model, a.seed_corpus, a.k, method=a.method, n_checkpoints=a.checkpoints,
                proj_dim=a.proj_dim, max_tokens=a.max_tokens, labels=a.labels)
    if a.out: r.save(a.out)
    d = r.to_dict(with_texts=False)
    if a.reference:
        d["vocab_containment"] = vocab_containment([r.texts[i] for i in r.indices], load_corpus(a.reference))
    _out(d) if a.json else print("\n".join(f"{x['rank']:>4}  {x['score']:+.4f}  {x['text'][:100]}" for x in r.selected()))


def cmd_serve(a):
    from .web.server import serve
    serve(a.root, host=a.host, port=a.port, token=a.token, readonly=a.readonly)


def cmd_make_test_model(a):
    from .testmodel import build_tiny_model, finetune_head, topic_corpus
    print(build_tiny_model(a.out, n_layers=a.layers, untied_head=a.finetune_demo))
    if a.finetune_demo:
        stem = a.out[:-5] if a.out.endswith(".gguf") else a.out
        print(finetune_head(a.out, stem + ".ft.gguf", topic_corpus(40, "A", seed=100)))
        with open(stem + ".seed.txt", "w") as fh:
            fh.write("\n".join(topic_corpus(60, "A", seed=1) + topic_corpus(60, "B", seed=2)) + "\n")
        print(stem + ".seed.txt  (first 60 lines = topic A, the fine-tune topic)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("gguf-hack", description="GGUF inspection, patching, steering and evaluation")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true"); p.add_argument("--log-file")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help):
        sp = sub.add_parser(name, help=help); sp.set_defaults(fn=fn); sp.add_argument("--model", required=True); return sp

    s = add("inspect", cmd_inspect, "dump header, KV and tensor offsets")
    s.add_argument("--tensor"); s.add_argument("--json", action="store_true"); s.add_argument("--kv", action="store_true")
    s = add("unpack-scales", cmd_unpack, "export block scales to .npy/.csv")
    s.add_argument("--tensor", required=True); s.add_argument("--out", required=True); s.add_argument("--field", default="d")
    s = add("patch-scale", cmd_patch, "edit FP16 block scales in place")
    s.add_argument("--tensor", required=True, help="name or glob"); s.add_argument("--field", default="d")
    s.add_argument("--scale-map"); s.add_argument("--multiplier", type=float); s.add_argument("--set", dest="set_to", type=float)
    s.add_argument("--dry-run", action="store_true"); s.add_argument("--no-backup", action="store_true"); s.add_argument("--journal")
    s = add("restore", cmd_restore, "restore model from its .bak")
    s = add("steer", cmd_steer, "inject a steering vector (numpy backend) or export a llama.cpp control vector")
    s.add_argument("--prompt", default="Hello"); s.add_argument("--vector", required=True); s.add_argument("--layer", type=int, required=True)
    s.add_argument("--alpha", type=float, default=1.0); s.add_argument("--tokens", type=int, default=16); s.add_argument("--export")
    s = add("eval", cmd_eval, "regression check vs baseline")
    s.add_argument("--baseline"); s.add_argument("--save-baseline"); s.add_argument("--eval-dataset")
    s.add_argument("--max-ppl-ratio", type=float, default=1.5); s.add_argument("--max-kl", type=float, default=0.5)
    s = add("hexdump", cmd_hexdump, "annotated hex view of file/tensor bytes")
    s.add_argument("--offset", type=lambda x: int(x, 0)); s.add_argument("--tensor"); s.add_argument("--block", type=int, default=0)
    s.add_argument("--length", type=int, default=256)
    s = add("poke", cmd_poke, "write raw bytes (hex editor)")
    s.add_argument("--offset", type=lambda x: int(x, 0)); s.add_argument("--tensor"); s.add_argument("--block", type=int, default=0)
    s.add_argument("--hex", required=True); s.add_argument("--dry-run", action="store_true"); s.add_argument("--no-backup", action="store_true")
    s = add("viz", cmd_viz, "render plots / HTML report")
    s.add_argument("kind", choices=["bytemap", "hist", "heatmap", "layers", "report"]); s.add_argument("--tensor")
    s.add_argument("--field", default="d"); s.add_argument("--out", required=True)
    s = sub.add_parser("harness", help="run an experiment spec (JSON/YAML)"); s.set_defaults(fn=cmd_harness); s.add_argument("spec")
    s = add("recover-data", cmd_recover, "approximate fine-tuning data from a base/fine-tuned pair (SELECT)")
    s.add_argument("--base", required=True, help="base model; --model is the fine-tuned one")
    s.add_argument("--seed-corpus", required=True, help=".txt / .jsonl / .json of candidate texts")
    s.add_argument("--k", type=int, default=100); s.add_argument("--method", default="select", choices=["select", "topk", "loss_drop", "random"])
    s.add_argument("--checkpoints", type=int, default=3); s.add_argument("--proj-dim", type=int, default=4096)
    s.add_argument("--max-tokens", type=int, default=64); s.add_argument("--labels", default="pseudo", choices=["pseudo", "text"])
    s.add_argument("--out", help="write selected examples as .jsonl"); s.add_argument("--reference", help="true data, for vocab containment")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("serve", help="start the web dashboard"); s.set_defaults(fn=cmd_serve)
    s.add_argument("--root", default="."); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token"); s.add_argument("--readonly", action="store_true")
    s = sub.add_parser("make-test-model", help="write a tiny synthetic GGUF"); s.set_defaults(fn=cmd_make_test_model)
    s.add_argument("out"); s.add_argument("--layers", type=int, default=4)
    s.add_argument("--finetune-demo", action="store_true",
                   help="also write <out>.ft.gguf (head fine-tuned on topic A) and <out>.seed.txt for recover-data")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    configure("DEBUG" if a.verbose else "INFO", a.log_file)
    try:
        a.fn(a)
    except (KeyError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
