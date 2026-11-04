"""HTTP API + dashboard server (standard library only).

Security model
--------------
* binds to ``127.0.0.1`` by default and rejects requests whose ``Host`` is not local/allowed
  (blocks DNS-rebinding);
* only ``.gguf`` files **inside ``--root``** are reachable (no path traversal);
* mutating endpoints (POST) require the custom header ``X-GGUF-Hack: 1`` (so other web pages can't
  forge them) and are disabled entirely with ``--readonly``;
* optional bearer ``--token`` (``Authorization: Bearer ..`` or ``?token=..``).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import viz
from ..evaluator import check_thresholds, evaluate, load_suite
from ..harness import Harness
from ..hexedit import block_view, hex_view, parse_hex, search_bytes, tensor_start
from ..inspector import GGUFReader, open_gguf
from ..log import get_logger
from ..patcher import PatchError, Patcher, restore_backup
from ..quant_codec import block_stats
from ..tracer import NumpyLlama

log = get_logger("server")
STATIC = os.path.join(os.path.dirname(__file__), "static")
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml",
        ".png": "image/png", ".json": "application/json"}


def clean(o):
    """JSON-safe: numpy -> python, NaN/Inf -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    return o


class HttpError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg); self.code, self.msg = code, msg


class Session:
    """Open model + patcher + (lazy) reference model, guarded by a lock."""

    def __init__(self, path: str, readonly: bool):
        self.path, self.readonly = path, readonly
        self.reader: GGUFReader = open_gguf(path, writable=not readonly)
        self.patcher = None if readonly else Patcher(self.reader, backup=True, journal=True)
        self.lock = threading.RLock()
        self._model = None
        self.baseline = None

    @property
    def model(self):
        if self._model is None:
            self._model = NumpyLlama(self.reader)
        return self._model

    def invalidate(self):
        if self._model: self._model.invalidate()


class App:
    def __init__(self, root, token=None, readonly=False):
        self.root = os.path.realpath(root)
        self.token, self.readonly = token, readonly
        self.sessions: dict[str, Session] = {}
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ---- paths / sessions ------------------------------------------------------
    def resolve(self, rel: str) -> str:
        p = os.path.realpath(os.path.join(self.root, rel))
        if os.path.commonpath([p, self.root]) != self.root or not p.endswith(".gguf"):
            raise HttpError(403, "model path must be a .gguf inside the server root")
        if not os.path.isfile(p):
            raise HttpError(404, f"no such model: {rel}")
        return p

    def resolve_corpus(self, rel: str) -> str:
        p = os.path.realpath(os.path.join(self.root, rel))
        if os.path.commonpath([p, self.root]) != self.root or not p.endswith((".txt", ".jsonl", ".json")):
            raise HttpError(403, "corpus must be a .txt/.jsonl/.json inside the server root")
        if not os.path.isfile(p):
            raise HttpError(404, f"no such corpus: {rel}")
        return p

    def list_corpora(self):
        out = []
        for dp, dn, fn in os.walk(self.root):
            dn[:] = [d for d in dn if not d.startswith(".") and d != "runs"]
            for f in fn:
                if f.endswith((".txt", ".jsonl", ".json")) and not f.endswith(".bak"):
                    full = os.path.join(dp, f); out.append({"path": os.path.relpath(full, self.root), "size": os.path.getsize(full)})
        return sorted(out, key=lambda x: x["path"])

    def session(self, rel: str) -> Session:
        p = self.resolve(rel)
        with self._lock:
            if p not in self.sessions:
                self.sessions[p] = Session(p, self.readonly)
            return self.sessions[p]

    def list_models(self):
        out = []
        for dp, dn, fn in os.walk(self.root):
            dn[:] = [d for d in dn if not d.startswith(".") and d != "runs"]
            if dp[len(self.root):].count(os.sep) > 4: dn[:] = []
            for f in fn:
                if f.endswith(".gguf"):
                    full = os.path.join(dp, f)
                    out.append({"path": os.path.relpath(full, self.root), "size": os.path.getsize(full)})
        return sorted(out, key=lambda x: x["path"])

    # ---- GET handlers ----------------------------------------------------------
    def get(self, route: str, q: dict):
        a = lambda k, d=None: (q.get(k) or [d])[0]
        if route == "/api/info":
            return {"root": self.root, "readonly": self.readonly, "version": __import__("gguf_weight_hacker").__version__}
        if route == "/api/models":
            return self.list_models()
        if route == "/api/corpora":
            return self.list_corpora()
        if route == "/api/job":
            j = self.jobs.get(a("id"))
            if not j: raise HttpError(404, "no such job")
            return j
        s = self.session(a("model", ""))
        r = s.reader
        with s.lock:
            if route == "/api/summary":
                d = r.summary(); d["kv"] = {k: e.summary() for k, e in r.kv.items()}
                d["journal"] = len(s.patcher.journal) if s.patcher else 0
                d["has_backup"] = os.path.exists(s.path + ".bak")
                return d
            if route == "/api/tensors":
                import fnmatch
                pat = a("q", "") or ""
                ts = [t for t in r if not pat or fnmatch.fnmatchcase(t.name, pat if any(c in pat for c in "*?[") else f"*{pat}*")]
                return [{"name": t.name, "type": t.type.name, "shape": list(t.shape), "offset": t.offset_bytes,
                         "size": t.size_bytes, "blocks": t.blocks_count, "layer": t.layer_index,
                         "scalable": bool(t.layout.scale_fields)} for t in ts]
            if route == "/api/tensor":
                t = self._tensor(r, a("name"))
                d = t.to_dict(); d["stats"] = block_stats(r, t.name) if t.layout.scale_fields else None
                return d
            if route == "/api/scales":
                return self._scales(r, self._tensor(r, a("name")), a("field", "d"), int(a("bins", 60)))
            if route == "/api/layers":
                return self._layers(r, a("kind", "ffn_down"), a("field", "d"))
            if route == "/api/hex":
                return hex_view(r, int(a("offset", 0), 0), min(int(a("length", 256)), 4096), 16)
            if route == "/api/block":
                return block_view(r, a("tensor"), int(a("block", 0)))
            if route == "/api/locate":
                t = self._tensor(r, a("tensor")); return {"offset": tensor_start(r, t.name, int(a("block", 0)))}
            if route == "/api/find":
                return {"hits": search_bytes(r, parse_hex(a("hex", "")), int(a("start", 0)))}
            if route == "/api/journal":
                return [x.to_dict() | {"before": x.before.hex(), "after": x.after.hex()} for x in (s.patcher.journal if s.patcher else [])][-200:]
        raise HttpError(404, f"no route {route}")

    def get_png(self, route: str, q: dict) -> bytes:
        a = lambda k, d=None: (q.get(k) or [d])[0]
        s = self.session(a("model", "")); r = s.reader; kind = a("kind")
        with s.lock:
            if kind == "bytemap": return viz.byte_map(r)
            if kind == "hist": return viz.scale_histogram(r, a("tensor"), a("field", "d"))
            if kind == "heatmap": return viz.scale_heatmap(r, a("tensor"), a("field", "d"))
            if kind == "layers": return viz.layer_stats(r, a("tensor", "ffn_down"), a("field", "d"))
        raise HttpError(400, "unknown plot kind")

    @staticmethod
    def _tensor(r, name):
        if name not in r.tensors: raise HttpError(404, f"no tensor {name!r}")
        return r.tensors[name]

    @staticmethod
    def _scales(r, t, field, bins):
        if not t.layout.scale_fields or field not in [f.name for f in t.layout.scale_fields]:
            raise HttpError(400, f"{t.name} has no fp16 scale field {field!r}")
        s = r.scales(t.name, field); fin = s[np.isfinite(s)]
        hist, edges = np.histogram(fin, bins=bins)
        bpr = max(1, t.shape[0] // t.layout.block_elems)
        grid = s[: (s.size // bpr) * bpr].reshape(-1, bpr)
        # downsample for the heatmap: block-mean to at most 128 x 128
        sr, sc = max(1, grid.shape[0] // 128), max(1, grid.shape[1] // 128)
        gr, gc = grid.shape[0] // sr, grid.shape[1] // sc
        small = np.nanmean(grid[: gr * sr, : gc * sc].reshape(gr, sr, gc, sc), axis=(1, 3))
        return {"field": field, "n": int(s.size), "min": fin.min() if fin.size else None, "max": fin.max() if fin.size else None,
                "mean": fin.mean() if fin.size else None, "hist": hist, "edges": edges,
                "heat": {"rows": gr, "cols": gc, "data": np.round(small, 8), "row_step": sr, "col_step": sc}}

    @staticmethod
    def _layers(r, kind, field):
        out = []
        for l in range(r.num_layers()):
            n = f"blk.{l}.{kind}.weight"
            if n in r.tensors and any(f.name == field for f in r.tensors[n].layout.scale_fields):
                s = r.scales(n, field); out.append({"layer": l, "mean": s.mean(), "std": s.std(), "max": s.max(), "min": s.min()})
        return out

    # ---- POST handlers -----------------------------------------------------------
    def post(self, route: str, body: dict):
        if route == "/api/recover":            # read-only analysis: allowed in --readonly mode
            return self._start_recover(body)
        if self.readonly:
            raise HttpError(403, "server is read-only")
        s = self.session(body.get("model", ""))
        r = s.reader
        try:
            with s.lock:
                if route == "/api/poke":
                    res = s.patcher.poke_bytes(int(body["offset"]), parse_hex(body["hex"]), dry_run=bool(body.get("dry_run")))
                    if not body.get("dry_run"): s.invalidate()
                    return res.to_dict()
                if route == "/api/patch":
                    t = self._tensor(r, body["tensor"])
                    kw = {k: body[k] for k in ("multiplier", "set_to") if body.get(k) is not None}
                    res = s.patcher.patch_scale(t.name, field=body.get("field", "d"), dry_run=bool(body.get("dry_run")), **kw)
                    if not body.get("dry_run"): s.invalidate()
                    d = res.to_dict(); d["records"] = len(d["records"]); return d
                if route == "/api/revert":
                    n = s.patcher.revert(body.get("n")); s.invalidate(); return {"reverted": n}
                if route == "/api/restore":
                    s.reader.close(); restore_backup(s.path)
                    with self._lock: self.sessions.pop(s.path, None)
                    return {"restored": True}
                if route == "/api/eval":
                    m = s.model; suite = load_suite(None)
                    res = evaluate(m, suite, s.baseline[1] if (s.baseline and not body.get("set_baseline")) else None)
                    if body.get("set_baseline") or s.baseline is None:
                        s.baseline = (res.to_dict(), res.logprobs)
                    d = res.to_dict(); d["baseline_ppl"] = s.baseline[0]["perplexity"]
                    d["verdict"] = check_thresholds(res, s.baseline[0]); return d
                if route == "/api/steer_norms":
                    return {"norms": s.model.activation_norms(body.get("text", "Hello world"))}
        except (PatchError, KeyError, ValueError, IndexError) as e:
            raise HttpError(400, str(e))
        if route == "/api/sweep":
            return self._start_sweep(s, body)
        raise HttpError(404, f"no route {route}")

    def _start_recover(self, b: dict):
        from ..datarecover import recover
        base, ft = self.resolve(b.get("base", "")), self.resolve(b.get("finetuned", ""))
        corpus = self.resolve_corpus(b.get("corpus", ""))
        if base == ft: raise HttpError(400, "pick two different models")
        k = max(1, min(int(b.get("k", 50)), 5000)); method = b.get("method", "select")
        if method not in ("select", "topk", "loss_drop", "random"): raise HttpError(400, "bad method")
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "kind": "recover", "status": "running", "result": None, "error": None, "started": time.time()}
        self.jobs[jid] = job
        for p in (base, ft):                   # flush any pending in-dashboard edits first
            sess = self.sessions.get(p)
            if sess:
                with sess.lock: sess.reader.mmap.flush()

        def run():
            try:
                r = recover(base, ft, corpus, k, method=method, n_checkpoints=int(b.get("checkpoints", 3)),
                            labels=b.get("labels", "pseudo"), max_tokens=int(b.get("max_tokens", 64)))
                d = r.to_dict(); d["all_scores"] = r.all_scores
                for row in d["selected"]: row["text"] = row["text"][:400]
                job["result"], job["status"] = d, "done"
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "error", str(e)
        threading.Thread(target=run, daemon=True).start()
        return {"id": jid}

    def _start_sweep(self, s: Session, b: dict):
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "status": "running", "done": 0, "total": 0, "rows": [], "error": None, "started": time.time()}
        self.jobs[jid] = job
        values = [float(v) for v in b.get("values", [])]
        if not values: raise HttpError(400, "values required")

        def run():
            try:
                work = os.path.join(self.root, ".gguf-hack-runs", jid)
                with s.lock: s.reader.mmap.flush()
                with Harness(s.path, workdir=work) as h:
                    h.baseline()
                    groups = [[t] for t in h._resolve(b["pattern"])] if b.get("per_layer") else [h._resolve(b["pattern"])]
                    job["total"] = len(groups) * len(values)
                    for g in groups:
                        for v in values:
                            h.trial(g, **{b.get("param", "multiplier"): v}); job["done"] += 1
                            job["rows"] = h.table()
                    job["baseline"] = h.base.to_dict()
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "error", str(e)
        threading.Thread(target=run, daemon=True).start()
        return {"id": jid}


class Handler(BaseHTTPRequestHandler):
    app: App
    server_version = "gguf-hack"

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(clean(obj)).encode())

    def _guard(self, api: bool = True) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1") and host not in getattr(self.server, "allowed_hosts", ()):
            self._json(403, {"error": "bad Host header"}); return False
        tok = self.app.token
        if tok and api:
            qs = parse_qs(urlparse(self.path).query)
            given = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip() or (qs.get("token") or [""])[0]
            import hmac
            if not hmac.compare_digest(given, tok):
                self._json(401, {"error": "token required"}); return False
        return True

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/"):
            if not self._guard(): return
            try:
                q = parse_qs(u.query)
                if u.path == "/api/plot":
                    return self._send(200, self.app.get_png(u.path, q), "image/png")
                self._json(200, self.app.get(u.path, q))
            except HttpError as e: self._json(e.code, {"error": e.msg})
            except Exception as e:  # noqa: BLE001
                log.exception("GET %s failed", u.path); self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        if not self._guard(api=False): return
        name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC, name))
        if not full.startswith(os.path.realpath(STATIC) + os.sep) or not os.path.isfile(full):
            return self._json(404, {"error": "not found"})
        with open(full, "rb") as fh:
            self._send(200, fh.read(), MIME.get(os.path.splitext(full)[1], "application/octet-stream"))

    def do_POST(self):
        if not self._guard(): return
        if self.headers.get("X-GGUF-Hack") != "1":
            return self._json(403, {"error": "missing X-GGUF-Hack header"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000: raise HttpError(413, "body too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            self._json(200, self.app.post(urlparse(self.path).path, body))
        except HttpError as e: self._json(e.code, {"error": e.msg})
        except json.JSONDecodeError: self._json(400, {"error": "invalid JSON"})
        except Exception as e:  # noqa: BLE001
            log.exception("POST failed"); self._json(500, {"error": f"{type(e).__name__}: {e}"})


def make_server(root=".", host="127.0.0.1", port=8765, token=None, readonly=False) -> ThreadingHTTPServer:
    app = App(root, token, readonly)
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.app = app
    srv.allowed_hosts = (host,) if host not in ("0.0.0.0", "::") else ()
    return srv


def serve(root=".", host="127.0.0.1", port=8765, token=None, readonly=False):
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise SystemExit("refusing to bind a non-local address without --token (this server can modify model files)")
    srv = make_server(root, host, port, token, readonly)
    log.info("dashboard on http://%s:%d  root=%s  readonly=%s  token=%s", host, port, srv.app.root, readonly, bool(token))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        srv.server_close()
