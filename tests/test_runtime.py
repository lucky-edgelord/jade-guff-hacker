import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from gguf_weight_hacker import open_gguf
from gguf_weight_hacker.evaluator import check_thresholds, evaluate, load_baseline, save_baseline, weight_drift
from gguf_weight_hacker.harness import Harness, load_results, run_spec
from gguf_weight_hacker.patcher import Patcher
from gguf_weight_hacker.tracer import NumpyLlama, Steering, contrast_vector, export_control_vector


def test_forward_deterministic_and_steering(model):
    with open_gguf(model) as r:
        m = NumpyLlama(r); ids = m.encode("hello world")
        a, b = m.forward(ids), m.forward(ids)
        assert np.array_equal(a, b) and np.isfinite(a).all()
        v = contrast_vector(m, ["good"], ["bad"], layer=1)
        assert abs(np.linalg.norm(v) - 1) < 1e-5
        assert not np.allclose(m.forward(ids, steer=Steering(v, 1, 5.0)), a)
        assert np.allclose(m.forward(ids, steer=Steering(v, 1, 0.0)), a)   # alpha 0 is identity
        _, cap = m.forward(ids, capture=True)
        assert set(cap) == {0, 1, 2, 3}


def test_control_vector_export_is_valid_gguf(model, tmp_path):
    gguf = pytest.importorskip("gguf")
    p = export_control_vector(str(tmp_path / "cv.gguf"), {1: np.ones(256, np.float32)})
    r = gguf.GGUFReader(p)
    assert r.get_field("general.architecture").contents() == "controlvector"
    assert [t.name for t in r.tensors] == ["direction.2"]


def test_eval_baseline_roundtrip_and_regression(model, tmp_path):
    with open_gguf(model) as r:
        m = NumpyLlama(r); base = evaluate(m)
        save_baseline(base, str(tmp_path / "b.json"))
        meta, lps = load_baseline(str(tmp_path / "b.json"))
        same = evaluate(m, baseline_logprobs=lps)
        assert same.kl_mean < 1e-4 and same.top1_agree > 0.99
        assert check_thresholds(same, meta)["pass"]


def test_patch_changes_eval_and_drift(model, tmp_path):
    import shutil
    orig = str(tmp_path / "orig.gguf"); shutil.copy(model, orig)
    with open_gguf(model, writable=True) as rw, open_gguf(orig) as ro:
        base = evaluate(NumpyLlama(ro))
        Patcher(rw, backup=False).patch_scale("blk.2.ffn_down.weight", multiplier=1.5)
        res = evaluate(NumpyLlama(rw), baseline_logprobs=base.logprobs)
        assert res.kl_mean > 0
        d = weight_drift(ro, rw)
        assert [x["tensor"] for x in d] == ["blk.2.ffn_down.weight"] and abs(d[0]["rel_l2"] - 0.5) < 0.02


def test_harness_reverts_and_never_touches_original(model, tmp_path):
    import hashlib
    h0 = hashlib.md5(open(model, "rb").read()).hexdigest()
    with Harness(model, workdir=str(tmp_path / "w")) as h:
        h.baseline()
        h.trial("blk.1.ffn_down.weight", multiplier=1.5)
        bad = h.trial("blk.0.attn_q.weight", multiplier=1e9)      # rejected, must still revert cleanly
        assert bad.error and not bad.verdict["pass"]
        rows = h.sweep("blk.*.ffn_down.weight", "multiplier", [0.5, 2.0], per_layer=True)
        assert len(rows) == 8
        h.steer_sweep(contrast_vector(h.model, ["a"], ["b"], 1), [1], [1.0])
        path = h.save()
    assert hashlib.md5(open(model, "rb").read()).hexdigest() == h0
    base, rows = load_results(path)
    assert base and len(rows) == 2 + 8 + 1


def test_run_spec(model, tmp_path):
    spec = {"model": model, "workdir": str(tmp_path / "w"),
            "experiments": [{"tensors": "blk.0.ffn_*.weight", "param": "multiplier", "values": [0.9, 1.1]}]}
    h = run_spec(spec); assert len(h.trials) == 2; h.close()


# ---- web server ---------------------------------------------------------------------
@pytest.fixture
def srv(model, tmp_path):
    import os, shutil
    from gguf_weight_hacker.web.server import make_server
    root = tmp_path / "root"; root.mkdir(); shutil.copy(model, root / "m.gguf")
    (tmp_path / "secret.gguf").write_bytes(b"x")
    s = make_server(str(root), port=0); threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}", s
    s.shutdown(); s.server_close()


def _req(base, path, body=None, headers=None, host=None):
    h = dict(headers or {})
    if host: h["Host"] = host
    if body is not None: h.setdefault("X-GGUF-Hack", "1")
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None, h)
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_api_and_edit_cycle(srv):
    base, _ = srv
    assert _req(base, "/api/models")[1][0]["path"] == "m.gguf"
    assert _req(base, "/api/summary?model=m.gguf")[1]["n_tensors"] == 38
    _, loc = _req(base, "/api/locate?model=m.gguf&tensor=blk.0.attn_q.weight&block=1")
    code, res = _req(base, "/api/poke", {"model": "m.gguf", "offset": loc["offset"], "hex": "003c"})
    assert code == 200
    assert _req(base, "/api/block?model=m.gguf&tensor=blk.0.attn_q.weight&block=1")[1]["fields"][0]["values"] == [1.0]
    assert _req(base, "/api/revert", {"model": "m.gguf"})[1]["reverted"] == 1
    code, err = _req(base, "/api/poke", {"model": "m.gguf", "offset": 4, "hex": "ff"})
    assert code == 400 and "header" in err["error"]


def test_server_security(srv):
    base, s = srv
    assert _req(base, "/api/summary?model=../secret.gguf")[0] == 403          # traversal
    assert _req(base, "/api/summary?model=/etc/passwd")[0] == 403
    assert _req(base, "/api/summary?model=m.gguf", host="evil.example.com")[0] == 403   # DNS rebinding
    assert _req(base, "/api/patch", {"model": "m.gguf"}, headers={"X-GGUF-Hack": "0"})[0] == 403   # CSRF header
    assert _req(base, "/../../etc/passwd")[0] == 404


def test_server_readonly_and_token(model, tmp_path):
    import shutil
    from gguf_weight_hacker.web.server import make_server
    root = tmp_path / "r"; root.mkdir(); shutil.copy(model, root / "m.gguf")
    s = make_server(str(root), port=0, token="s3cret", readonly=True)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{s.server_address[1]}"
    try:
        assert _req(base, "/api/models")[0] == 401
        assert _req(base, "/api/models", headers={"Authorization": "Bearer s3cret"})[0] == 200
        assert _req(base, "/api/revert", {"model": "m.gguf"}, headers={"Authorization": "Bearer s3cret"})[0] == 403
        assert urllib.request.urlopen(base + "/app.js").status == 200       # static assets need no token
    finally:
        s.shutdown(); s.server_close()


def test_refuses_public_bind_without_token():
    from gguf_weight_hacker.web.server import serve
    with pytest.raises(SystemExit):
        serve(".", host="0.0.0.0")


def test_cli_smoke(model, capsys, tmp_path):
    from gguf_weight_hacker.cli import main
    main(["inspect", "--model", model, "--tensor", "blk.0.attn_q.weight", "--json"])
    assert json.loads(capsys.readouterr().out)["tensor_name"] == "blk.0.attn_q.weight"
    main(["patch-scale", "--model", model, "--tensor", "blk.1.ffn_down.weight", "--multiplier", "2", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    main(["hexdump", "--model", model, "--tensor", "blk.0.attn_q.weight", "--length", "32"])
    assert "attn_q.weight#0.d" in capsys.readouterr().out
    main(["unpack-scales", "--model", model, "--tensor", "blk.1.ffn_down.weight", "--out", str(tmp_path / "s.npy")])
    assert np.load(tmp_path / "s.npy").size == 4096
    main(["viz", "bytemap", "--model", model, "--out", str(tmp_path / "m.png")])
    assert (tmp_path / "m.png").read_bytes()[1:4] == b"PNG"
    with pytest.raises(SystemExit):
        main(["inspect", "--model", model, "--tensor", "nope"])
