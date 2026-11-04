import json
import threading
import time
import urllib.request

import pytest

from gguf_weight_hacker.datarecover import load_corpus, recover, vocab_containment
from gguf_weight_hacker.testmodel import build_tiny_model, finetune_head, topic_corpus


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("pair")
    base = build_tiny_model(str(d / "base.gguf"), untied_head=True, seed=5)
    ft = finetune_head(base, str(d / "base.ft.gguf"), topic_corpus(40, "A", seed=100))   # hidden fine-tune set
    corpus = topic_corpus(40, "A", seed=1) + topic_corpus(40, "B", seed=2)                # public seed pool
    (d / "seed.txt").write_text("\n".join(corpus) + "\n")
    return d, base, ft, corpus


def _purity(r, n_a=40):
    return sum(i < n_a for i in r.indices) / len(r.indices)


def test_select_recovers_finetune_topic(pair):
    _, base, ft, corpus = pair
    r = recover(base, ft, corpus, k=15, method="select")
    rnd = recover(base, ft, corpus, k=15, method="random", seed=3)
    assert _purity(r) >= 0.7 and _purity(r) > _purity(rnd)
    assert r.info["head_changed"] and r.info["changed_tensors"] == 1
    assert len(set(r.indices)) == 15                       # no duplicates
    assert all(b >= -1 and b <= 1 for b in r.gains)


@pytest.mark.parametrize("method", ["topk", "loss_drop"])
def test_baselines_beat_random(pair, method):
    _, base, ft, corpus = pair
    assert _purity(recover(base, ft, corpus, k=15, method=method)) >= 0.6


def test_identical_models_rejected(pair):
    _, base, _, corpus = pair
    with pytest.raises(ValueError, match="identical"):
        recover(base, base, corpus, k=5)


def test_mismatched_models_rejected(pair, tmp_path):
    _, base, _, corpus = pair
    other = build_tiny_model(str(tmp_path / "o.gguf"), n_layers=2)
    with pytest.raises(ValueError):
        recover(base, other, corpus, k=5)


def test_corpus_formats_and_metric(tmp_path):
    (tmp_path / "a.txt").write_text("one two\n\nthree\n")
    (tmp_path / "a.jsonl").write_text('{"text": "x y"}\n{"text": "z"}\n')
    (tmp_path / "a.json").write_text(json.dumps(["p", {"text": "q"}]))
    assert load_corpus(str(tmp_path / "a.txt")) == ["one two", "three"]
    assert load_corpus(str(tmp_path / "a.jsonl")) == ["x y", "z"]
    assert load_corpus(str(tmp_path / "a.json")) == ["p", "q"]
    assert vocab_containment(["a b"], ["a b c d"]) == 1.0
    assert vocab_containment(["a"], ["b"]) == 0.0


def test_save_and_cli(pair, tmp_path, capsys):
    from gguf_weight_hacker.cli import main
    d, base, ft, _ = pair
    out = tmp_path / "sel.jsonl"
    main(["recover-data", "--base", base, "--model", ft, "--seed-corpus", str(d / "seed.txt"), "--k", "5",
          "--out", str(out), "--json"])
    res = json.loads(capsys.readouterr().out)
    assert res["k"] == 5 and len(out.read_text().splitlines()) == 5


def test_web_recover_job(pair):
    from gguf_weight_hacker.web.server import make_server
    d, *_ = pair
    s = make_server(str(d), port=0, readonly=True)           # analysis is allowed read-only
    threading.Thread(target=s.serve_forever, daemon=True).start()
    B = f"http://127.0.0.1:{s.server_address[1]}"
    try:
        assert [c["path"] for c in json.load(urllib.request.urlopen(B + "/api/corpora"))] == ["seed.txt"]
        req = urllib.request.Request(B + "/api/recover", json.dumps({"base": "base.gguf", "finetuned": "base.ft.gguf",
                                     "corpus": "seed.txt", "k": 10}).encode(), {"X-GGUF-Hack": "1"})
        jid = json.load(urllib.request.urlopen(req))["id"]
        for _ in range(200):
            j = json.load(urllib.request.urlopen(B + "/api/job?id=" + jid))
            if j["status"] != "running": break
            time.sleep(0.2)
        assert j["status"] == "done", j["error"]
        assert j["result"]["k"] == 10 and len(j["result"]["all_scores"]) == 80
        bad = urllib.request.Request(B + "/api/recover", json.dumps({"base": "base.gguf", "finetuned": "base.ft.gguf",
                                     "corpus": "../../etc/passwd"}).encode(), {"X-GGUF-Hack": "1"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 403
    finally:
        s.shutdown(); s.server_close()
