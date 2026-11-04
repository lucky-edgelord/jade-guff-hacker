"use strict";
/* GGUF Vault dashboard — vanilla JS, no dependencies.
   All model-derived strings are inserted with textContent (never innerHTML): GGUF files are untrusted input. */

const $ = (s, r = document) => r.querySelector(s);
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") throw new Error("html not allowed");
    else if (v === true) e.setAttribute(k, "");
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const svgIcon = d => { const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); s.setAttribute("viewBox", "0 0 24 24");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path"); p.setAttribute("d", d); s.append(p); return s; };

async function api(path, body) {
  const opt = { headers: {} };
  if (TOKEN) opt.headers.Authorization = "Bearer " + TOKEN;
  if (body) { opt.method = "POST"; opt.headers["X-GGUF-Hack"] = "1"; opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const q = o => new URLSearchParams(Object.fromEntries(Object.entries(o).filter(([, v]) => v != null && v !== ""))).toString();
function toast(msg, err) { const t = h("div", { class: "toast" + (err ? " err" : ""), role: "status" }, msg); document.body.append(t); setTimeout(() => t.remove(), err ? 6000 : 3200); }
const fmtB = n => n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(1) + " KB" : n < 1073741824 ? (n / 1048576).toFixed(1) + " MB" : (n / 1073741824).toFixed(2) + " GB";
const fnum = (x, d = 4) => x == null ? "—" : Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : (+x.toFixed(d)).toString();
const hex = (n, w = 8) => n.toString(16).padStart(w, "0");

/* ───────────── canvas helpers ───────────── */
/* axis tick label with enough significant digits to distinguish neighbouring ticks */
function tick(v, range) {
  if (v === 0) return "0"; const dec = Math.max(0, 2 - Math.floor(Math.log10(Math.abs(range) || 1)));
  return (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && dec > 6)) ? v.toExponential(1) : (+v.toFixed(Math.min(dec, 8))).toString();
}
function canvas(w, hgt) {
  const c = h("canvas"); c.style.height = hgt + "px"; c._h = hgt; return c;
}
function prep(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth || 600, hh = c._h;
  c.width = w * dpr; c.height = hh * dpr; const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, hh);
  return { g, w, h: hh };
}
function axes(g, w, hh, m, xlab, ylab, ymin, ymax) {
  g.strokeStyle = css("--line"); g.fillStyle = css("--muted"); g.font = "11px " + css("--font-mono"); g.lineWidth = 1;
  g.beginPath(); for (let i = 0; i <= 4; i++) { const y = m.t + (hh - m.t - m.b) * i / 4; g.moveTo(m.l, y); g.lineTo(w - m.r, y);
    g.textAlign = "right"; g.fillText(tick(ymax - (ymax - ymin) * i / 4, ymax - ymin), m.l - 6, y + 4); } g.stroke();
  g.textAlign = "center"; if (xlab) g.fillText(xlab, (m.l + w - m.r) / 2, hh - 4);
}
function drawHist(c, hist, edges) {
  requestAnimationFrame(() => { const { g, w, h: hh } = prep(c), m = { l: 52, r: 10, t: 8, b: 26 };
    const mx = Math.max(...hist, 1); axes(g, w, hh, m, "scale value", "", 0, mx);
    const bw = (w - m.l - m.r) / hist.length; g.fillStyle = css("--jade-mid");
    hist.forEach((v, i) => { const bh = (hh - m.t - m.b) * v / mx; g.fillRect(m.l + i * bw + 1, hh - m.b - bh, Math.max(1, bw - 2), bh); });
    g.fillStyle = css("--muted"); g.textAlign = "left"; g.fillText(fnum(edges[0]), m.l, hh - 12); g.textAlign = "right"; g.fillText(fnum(edges.at(-1)), w - m.r, hh - 12); });
}
function ramp(t) { // dark green → jade → gold
  const a = [[8, 30, 24], [20, 120, 88], [63, 224, 165], [233, 207, 120]], x = Math.min(.999, Math.max(0, t)) * 3, i = Math.floor(x), f = x - i;
  return a[i].map((v, k) => Math.round(v + (a[i + 1][k] - v) * f));
}
function drawHeat(c, heat) {
  requestAnimationFrame(() => { const { g, w, h: hh } = prep(c); const { rows, cols, data } = heat; if (!rows || !cols) return;
    const flat = data.flat().filter(v => v != null), lo = Math.min(...flat), hi = Math.max(...flat), cw = w / cols, ch = hh / rows;
    data.forEach((row, y) => row.forEach((v, x) => { const [r, gg, b] = ramp(hi > lo ? (v - lo) / (hi - lo) : 0); g.fillStyle = `rgb(${r},${gg},${b})`; g.fillRect(x * cw, y * ch, cw + .5, ch + .5); })); });
}
function drawLine(c, series, xlab) { // series: [{name, color, pts:[[x,y],..], err:[..]?}]
  requestAnimationFrame(() => { const { g, w, h: hh } = prep(c), m = { l: 58, r: 14, t: 10, b: 40 };
    const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map((p, i) => [p[1] - (s.err?.[i] || 0), p[1] + (s.err?.[i] || 0)]).flat()).filter(v => v != null);
    if (!xs.length) return; let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys); if (x1 === x0) x1 += 1; if (y1 === y0) { y1 += 1; y0 -= 1; }
    const pad = (y1 - y0) * .08; y0 -= pad; y1 += pad; axes(g, w, hh, m, xlab, "", y0, y1);
    const X = x => m.l + (x - x0) / (x1 - x0) * (w - m.l - m.r), Y = y => m.t + (1 - (y - y0) / (y1 - y0)) * (hh - m.t - m.b);
    g.fillStyle = css("--muted"); g.textAlign = "center"; [x0, (x0 + x1) / 2, x1].forEach(x => g.fillText(fnum(x, 2), X(x), hh - 22));
    series.forEach(s => { g.strokeStyle = s.color; g.fillStyle = s.color; g.lineWidth = 1.6; g.beginPath();
      s.pts.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); g.stroke();
      s.pts.forEach((p, i) => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 3, 0, 7); g.fill();
        if (s.err?.[i]) { g.beginPath(); g.moveTo(X(p[0]), Y(p[1] - s.err[i])); g.lineTo(X(p[0]), Y(p[1] + s.err[i])); g.stroke(); } }); }); });
}
function drawByteMap(c, tensors, total, dataStart) {
  requestAnimationFrame(() => { const { g, w, h: hh } = prep(c), types = [...new Set(tensors.map(t => t.type))], pal = ["--jade", "--gold", "--info", "--violet", "--blue", "--danger", "--warn"];
    const col = {}; types.forEach((t, i) => col[t] = css(pal[i % pal.length])); const S = x => x / total * w;
    g.fillStyle = css("--line-strong"); g.fillRect(0, 0, Math.max(2, S(dataStart)), hh);
    tensors.forEach(t => { g.fillStyle = col[t.type]; g.globalAlpha = .85; g.fillRect(S(t.offset), 0, Math.max(1, S(t.size) - 1), hh); }); g.globalAlpha = 1; c._col = col; c.dispatchEvent(new Event("legend")); });
  return () => c._col;
}

/* ───────────── state ───────────── */
const S = { model: null, info: null, summary: null, tab: "overview", tensor: null, hex: { offset: 0, sel: 0 } };
const TABS = [
  ["overview", "Overview", "M3 12l9-8 9 8M5 10v9h14v-9"],
  ["tensors", "Tensors", "M4 6h16M4 12h16M4 18h10"],
  ["hex", "Hex editor", "M8 4l-5 8 5 8M16 4l5 8-5 8"],
  ["layers", "Layers", "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5"],
  ["lab", "Experiments", "M9 3v6L4 19a2 2 0 002 3h12a2 2 0 002-3l-5-10V3M8 3h8"],
];
const VIEWS = {};
async function refreshSummary() { S.summary = await api("/api/summary?" + q({ model: S.model })); }
function ro() { return S.info.readonly; }

/* ───────────── Overview ───────────── */
VIEWS.overview = async root => {
  const s = S.summary, tensors = await api("/api/tensors?" + q({ model: S.model }));
  const stat = (k, v, sub) => h("div", { class: "card stat" }, h("span", { class: "k" }, k), h("span", { class: "v" }, v), h("span", { class: "s" }, sub || ""));
  const bt = Object.entries(s.bytes_by_type).sort((a, b) => b[1] - a[1]), tot = s.size_bytes;
  const cv = canvas(0, 70), leg = h("div", { class: "legend" });
  const getcol = drawByteMap(cv, tensors, tot, s.data_start);
  cv.addEventListener("legend", () => { leg.replaceChildren(h("span", {}, h("i", { style: "background:var(--line-strong)" }), "header + metadata"),
    ...Object.entries(cv._col).map(([t, c]) => h("span", {}, h("i", { style: `background:${c}` }), t, " · ", fmtB(s.bytes_by_type[t] || 0)))); });
  const kv = h("dl", { class: "kv" }); Object.entries(s.kv).forEach(([k, v]) => kv.append(h("dt", {}, k), h("dd", {}, typeof v === "object" ? JSON.stringify(v) : String(v))));
  root.append(
    h("div", { class: "grid g4" }, stat("Architecture", s.architecture || "?", "GGUF v" + s.version), stat("Quantization", s.file_type.replace("MOSTLY_", ""), Object.keys(s.type_counts).join(" · ")),
      stat("Layers", s.n_layers, s.n_tensors + " tensors"), stat("File size", fmtB(s.size_bytes), "align " + s.alignment + " · data @ " + s.data_start.toLocaleString())),
    h("div", { class: "card" }, h("h2", {}, "Byte map", h("small", {}, "header → tensors")), cv, h("div", { style: "height:10px" }), leg),
    h("div", { class: "card" }, h("h2", {}, "Metadata", h("small", {}, Object.keys(s.kv).length + " keys")), h("div", { class: "scroll", style: "padding:14px;max-height:380px" }, kv)));
};

/* ───────────── Tensors ───────────── */
VIEWS.tensors = async root => {
  const all = await api("/api/tensors?" + q({ model: S.model })); const list = h("tbody"), side = h("div"), filt = h("input", { placeholder: "filter e.g. ffn_down or blk.3.*", style: "width:100%" });
  const draw = () => { const f = filt.value.trim().toLowerCase(); list.replaceChildren(...all.filter(t => !f || t.name.toLowerCase().includes(f.replace(/\*/g, ""))).slice(0, 600).map(t =>
    h("tr", { class: t.name === S.tensor ? "sel" : "", onclick: () => { S.tensor = t.name; draw(); detail(); } }, h("td", { class: "mono" }, t.name), h("td", {}, h("span", { class: "pill" }, t.type)), h("td", { class: "mono" }, t.shape.join("×"))))); };
  filt.addEventListener("input", draw);
  async function detail() {
    if (!S.tensor) { side.replaceChildren(h("div", { class: "empty" }, "Select a tensor")); return; }
    const t = await api("/api/tensor?" + q({ model: S.model, name: S.tensor }));
    const body = h("div", { class: "grid" });
    body.append(h("dl", { class: "kv" }, ...[["type", t.type], ["shape", t.shape.join(" × ")], ["offset", t.offset_bytes.toLocaleString() + "  (0x" + hex(t.offset_bytes, 1) + ")"], ["size", fmtB(t.size_bytes)], ["blocks", t.blocks_count.toLocaleString() + " × " + t.block_size_bytes + " B"],
      ["layout", t.fields.map(f => f.name + "@" + f.offset).join("  ")]].flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
    const fields = t.fields.filter(f => f.dtype === "<f2" && (f.role === "scale" || f.role === "min")).map(f => f.name);
    if (fields.length) {
      const sel = h("select", {}, fields.map(f => h("option", {}, f))), hc = canvas(0, 150), hm = canvas(0, 190), stats = h("div", { class: "muted mono" });
      const load = async () => { const d = await api("/api/scales?" + q({ model: S.model, name: S.tensor, field: sel.value }));
        stats.textContent = `n=${d.n.toLocaleString()}  min=${fnum(d.min)}  max=${fnum(d.max)}  mean=${fnum(d.mean)}`; drawHist(hc, d.hist, d.edges); drawHeat(hm, d.heat); };
      sel.addEventListener("change", load);
      const mult = h("input", { type: "number", step: "any", value: "1.1", style: "width:90px" }), dry = h("input", { type: "checkbox", checked: true }), out = h("div", { class: "mono muted" });
      const go = async dry_run => { try { const r = await api("/api/patch", { model: S.model, tensor: S.tensor, field: sel.value, multiplier: +mult.value, dry_run }); out.textContent =
        `${dry_run ? "dry run" : "applied"}: ${r.blocks_touched} blocks, mean ${fnum(r.stats.orig_mean)} → ${fnum(r.stats.new_mean)}, max ratio ${fnum(r.stats.max_ratio, 3)}×`;
        if (!dry_run) { toast("Patched " + S.tensor); await refreshSummary(); load(); } } catch (e) { out.textContent = ""; toast(e.message, true); } };
      body.append(h("div", {}, h("div", { class: "row", style: "margin-bottom:8px" }, h("b", {}, "Scale field"), sel, stats), hc, h("div", { class: "muted", style: "margin:12px 0 6px;font-size:12px" }, "Block-scale heatmap (rows × blocks per row)"), hm,
        h("div", { class: "card", style: "background:var(--bg);padding:12px 14px" }, h("div", { class: "row" }, h("b", {}, "Patch scales ×"), mult,
          h("button", { class: "btn", onclick: () => go(true) }, "Preview"), h("button", { class: "btn primary", disabled: ro(), onclick: () => go(false) }, "Apply"),
          h("button", { class: "btn", onclick: () => { S.hex.offset = t.offset_bytes; switchTab("hex"); } }, "Open in hex")), out)));
      requestAnimationFrame(load);
    }
    side.replaceChildren(h("div", { class: "card" }, h("h2", {}, S.tensor), body));
  }
  root.append(h("div", { class: "grid g-side" }, h("div", { class: "card" }, h("h2", {}, "Tensors", h("small", {}, all.length)), filt, h("div", { style: "height:10px" }),
    h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "name"), h("th", {}, "type"), h("th", {}, "shape"))), list))), side));
  draw(); detail();
};

/* ───────────── Hex editor ───────────── */
VIEWS.hex = async root => {
  const PAGE = 512, st = S.hex; st.length = PAGE; let view = null, edited = new Set();
  const grid = h("div", { class: "hex", tabindex: "0", "aria-label": "hex view" }), insp = h("div", { class: "card insp" }), blk = h("div", { class: "card" });
  const offIn = h("input", { value: "0x" + st.offset.toString(16), style: "width:130px", class: "mono", "aria-label": "offset" });
  const tensors = await api("/api/tensors?" + q({ model: S.model })), tsel = h("select", { style: "max-width:230px" }, h("option", { value: "" }, "jump to tensor…"), tensors.map(t => h("option", { value: t.name }, t.name)));
  const bIn = h("input", { type: "number", min: 0, value: 0, style: "width:80px", "aria-label": "block" }), findIn = h("input", { placeholder: "find hex e.g. 47 47 55 46", class: "mono", style: "width:190px" });
  const jump = o => { st.offset = Math.max(0, o - (o % 16)); st.sel = o; load(); };
  const load = async () => {
    view = await api("/api/hex?" + q({ model: S.model, offset: st.offset, length: PAGE })); offIn.value = "0x" + view.offset.toString(16);
    grid.replaceChildren(...view.rows.map(r => { const bytes = h("div", { class: "bytes" });
      for (let j = 0; j < r.hex.length / 2; j++) { const o = r.offset + j, b = h("span", { class: `b r-${r.roles[j]}${o === st.sel ? " sel" : ""}${edited.has(o) ? " edited" : ""}${j && r.tags[j] !== r.tags[j - 1] ? " blockstart" : ""}`, title: r.tags[j], "data-o": o, onclick: () => select(o) }, r.hex.substr(j * 2, 2)); bytes.append(b); }
      return h("div", { class: "hr" }, h("span", { class: "off" }, hex(r.offset)), bytes, h("span", { class: "asc" }, r.ascii)); }));
    inspect();
  };
  const byteAt = o => { for (const r of view.rows) if (o >= r.offset && o < r.offset + r.hex.length / 2) return parseInt(r.hex.substr((o - r.offset) * 2, 2), 16); return null; };
  const tagAt = o => { for (const r of view.rows) if (o >= r.offset && o < r.offset + r.hex.length / 2) return r.tags[o - r.offset]; return ""; };
  function select(o) { st.sel = o; grid.querySelectorAll(".b.sel").forEach(e => e.classList.remove("sel")); const e = grid.querySelector(`[data-o="${o}"]`); if (e) e.classList.add("sel"); inspect(); }
  async function inspect() {
    const o = st.sel, v = byteAt(o), nx = byteAt(o + 1), tag = tagAt(o); if (v == null) return;
    const f16 = nx == null ? null : (() => { const b = new DataView(new ArrayBuffer(2)); b.setUint8(0, v); b.setUint8(1, nx); const u = b.getUint16(0, true), s = u >> 15 ? -1 : 1, e = (u >> 10) & 31, m = u & 1023;
      return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); })();
    const inp = h("input", { class: "hexin", value: v.toString(16).padStart(2, "0"), maxlength: 23, "aria-label": "new bytes (hex)" }), dry = h("input", { type: "checkbox" });
    const apply = async () => { try { const r = await api("/api/poke", { model: S.model, offset: o, hex: inp.value, dry_run: dry.checked }); if (dry.checked) toast(`dry run OK — would write ${r.stats.n_bytes} byte(s) in ${r.tensor}`);
      else { for (let i = 0; i < r.stats.n_bytes; i++) edited.add(o + i); toast("Wrote " + r.stats.n_bytes + " byte(s)"); await refreshSummary(); await load(); } } catch (e) { toast(e.message, true); } };
    inp.addEventListener("keydown", e => { if (e.key === "Enter") apply(); });
    const undo = async () => { try { const r = await api("/api/revert", { model: S.model, n: 1 }); toast("Reverted " + r.reverted + " edit(s)"); edited.clear(); await refreshSummary(); load(); } catch (e) { toast(e.message, true); } };
    insp.replaceChildren(h("h2", {}, "Inspector", h("small", {}, "0x" + hex(o))), h("div", { class: "big" }, "0x" + v.toString(16).padStart(2, "0").toUpperCase(), h("span", { class: "muted", style: "font:13px var(--font-mono);margin-left:12px" }, v + " · " + (v > 127 ? v - 256 : v) + " · " + v.toString(2).padStart(8, "0"))),
      h("dl", { class: "kv", style: "margin:10px 0" }, h("dt", {}, "region"), h("dd", {}, tag), h("dt", {}, "fp16 (this+next)"), h("dd", {}, f16 == null ? "—" : String(+f16.toPrecision(6))), h("dt", {}, "offset"), h("dd", {}, o.toLocaleString())),
      h("div", { class: "muted", style: "font-size:12px;margin:8px 0 4px" }, "Write bytes (hex, multi-byte ok)"), inp,
      h("div", { class: "row", style: "margin-top:10px" }, h("button", { class: "btn primary", disabled: ro(), onclick: apply }, "Write"), h("label", { class: "row", style: "gap:6px" }, dry, "dry run"), h("button", { class: "btn", disabled: ro() || !S.summary.journal, onclick: undo }, "Undo last")),
      h("div", { class: "muted", style: "font-size:12px;margin-top:10px" }, ro() ? "Server is read-only." : "Header & metadata are protected. A .bak backup is made on first write."));
    const m = /^(.+)#(\d+)\.(.+)$/.exec(tag);
    if (m) { try { const b = await api("/api/block?" + q({ model: S.model, tensor: m[1], block: m[2] })); const dq = b.dequant, cv = dq ? canvas(0, 70) : null;
      blk.replaceChildren(h("h2", {}, "Block " + b.block, h("small", {}, b.type)), h("div", { class: "mono muted", style: "margin-bottom:8px;word-break:break-all" }, m[1]),
        ...b.fields.map(f => h("div", { class: "fieldrow" }, h("b", { class: "r-" + f.role }, f.name), h("div", {}, h("div", { class: "muted", style: "font-size:11px" }, `+${f.block_offset} · ${f.nbytes}B ${f.dtype}`), h("div", { class: "vals" }, f.values.slice(0, 16).map(x => typeof x === "number" ? +x.toPrecision(5) : x).join(", ") + (f.values.length > 16 ? " …" : ""))))),
        dq ? h("div", { class: "muted", style: "margin:10px 0 4px;font-size:12px" }, "Dequantized elements") : null, cv);
      if (cv) requestAnimationFrame(() => { const { g, w, h: hh } = prep(cv), mx = Math.max(...dq.map(Math.abs), 1e-9), bw = w / dq.length; dq.forEach((v, i) => { g.fillStyle = v >= 0 ? css("--jade") : css("--gold"); const bh = Math.abs(v) / mx * hh / 2; g.fillRect(i * bw, v >= 0 ? hh / 2 - bh : hh / 2, Math.max(1, bw - 1), bh); }); }); } catch { blk.replaceChildren(); } }
    else blk.replaceChildren(h("h2", {}, "Block"), h("div", { class: "empty" }, "Select a byte inside a tensor"));
  }
  const nav = (d) => { st.offset = Math.max(0, st.offset + d); st.sel = st.offset; load(); };
  offIn.addEventListener("keydown", e => { if (e.key === "Enter") jump(parseInt(offIn.value, offIn.value.startsWith("0x") ? 16 : 10) || 0); });
  tsel.addEventListener("change", async () => { if (!tsel.value) return; const r = await api("/api/locate?" + q({ model: S.model, tensor: tsel.value, block: +bIn.value })); jump(r.offset); });
  bIn.addEventListener("change", () => tsel.dispatchEvent(new Event("change")));
  findIn.addEventListener("keydown", async e => { if (e.key !== "Enter") return; try { const r = await api("/api/find?" + q({ model: S.model, hex: findIn.value, start: st.sel + 1 })); if (r.hits.length) jump(r.hits[0]); else toast("not found"); } catch (er) { toast(er.message, true); } });
  grid.addEventListener("keydown", e => { const k = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 16, ArrowUp: -16 }[e.key]; if (!k) return; e.preventDefault(); const n = Math.max(0, st.sel + k); if (n < view.offset || n >= view.offset + view.length) { st.offset = n - (n % 16) - (k < 0 ? PAGE - 16 : 0); st.offset = Math.max(0, st.offset); st.sel = n; load(); } else select(n); });
  const legend = h("div", { class: "legend" }, ...[["scale", "scale d"], ["min", "min / dmin"], ["subscale", "sub-scales"], ["hbits", "high bits"], ["payload", "quant payload"], ["header", "header"], ["kv", "metadata"], ["tensor_info", "tensor info"], ["padding", "padding"]].map(([r, l]) => h("span", {}, h("i", { style: `background:var(--role-${r})` }), l)));
  root.append(h("div", { class: "card" }, h("h2", {}, "Weight hex editor", h("small", {}, "block-aware")),
    h("div", { class: "row", style: "margin-bottom:12px" }, offIn, h("button", { class: "btn", onclick: () => nav(-PAGE) }, "◀ prev"), h("button", { class: "btn", onclick: () => nav(PAGE) }, "next ▶"), tsel, bIn, findIn), legend, h("div", { style: "height:10px" }), h("div", { class: "hexwrap" }, grid, h("div", { class: "grid" }, insp, blk))));
  await load();
};

/* ───────────── Layers ───────────── */
VIEWS.layers = async root => {
  const kinds = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"], sel = h("select", {}, kinds.map(k => h("option", { selected: k === "ffn_down" }, k))), cv = canvas(0, 300), info = h("div", { class: "muted mono" });
  const load = async () => { const d = await api("/api/layers?" + q({ model: S.model, kind: sel.value })); info.textContent = d.length ? d.length + " layers" : "no scale data for this kind";
    drawLine(cv, [{ name: sel.value, color: css("--jade"), pts: d.map(r => [r.layer, r.mean]), err: d.map(r => r.std) }, { name: "max", color: css("--gold"), pts: d.map(r => [r.layer, r.max]) }], "layer"); };
  sel.addEventListener("change", load); root.append(h("div", { class: "card" }, h("h2", {}, "Per-layer block scales", h("small", {}, "mean ± std (jade) · max (gold)")), h("div", { class: "row", style: "margin-bottom:10px" }, sel, info), cv)); load();
};

/* ───────────── Experiments ───────────── */
VIEWS.lab = async root => {
  const out = h("div", { class: "muted" }, "No baseline yet."), evalBtn = (label, set) => h("button", { class: "btn" + (set ? "" : " primary"), onclick: async e => { e.target.disabled = true;
    try { const r = await api("/api/eval", { model: S.model, set_baseline: set }); out.replaceChildren(h("span", {}, "perplexity ", h("b", { class: "ok" }, fnum(r.perplexity, 3)), "  (baseline ", fnum(r.baseline_ppl, 3), ")  "),
      r.kl_mean != null ? h("span", {}, "KL ", h("b", {}, fnum(r.kl_mean)), "  top-1 agree ", h("b", {}, fnum(r.top1_agree * 100, 1) + "%"), "  ") : "", h("span", { class: r.verdict.pass ? "ok" : "bad" }, r.verdict.pass ? "✔ within thresholds" : "✘ " + r.verdict.reasons.join("; "))); } catch (er) { toast(er.message, true); } e.target.disabled = false; } }, label);
  const pat = h("input", { value: "blk.*.ffn_down.weight", style: "width:240px", class: "mono" }), vals = h("input", { value: "0.5, 0.8, 1.2, 2", style: "width:170px" }), pl = h("input", { type: "checkbox", checked: true }), prog = h("div", { class: "bar", style: "width:220px;display:none" }, h("i", { style: "width:0%" })), chart = canvas(0, 280), table = h("div"), cap = h("div", { class: "muted", style: "margin-top:6px;font-size:12px" });
  let metric = "kl_mean"; const msel = h("select", {}, ["kl_mean", "perplexity", "top1_agree"].map(m => h("option", {}, m))); msel.addEventListener("change", () => { metric = msel.value; render(last); });
  let last = null; const pal = ["--jade", "--gold", "--info", "--violet", "--blue", "--danger", "--warn", "--jade-mid"];
  function render(job) { if (!job || !job.rows.length) return; const g = {}; job.rows.filter(r => r.value != null && r[metric] != null).forEach(r => (g[r.tensors] ||= []).push([r.value, r[metric]]));
    const names = Object.keys(g); drawLine(chart, names.map((n, i) => ({ name: n, color: css(pal[i % pal.length]), pts: g[n].sort((a, b) => a[0] - b[0]) })), "edit value"); cap.textContent = names.length + " series · " + job.rows.length + " trials";
    table.replaceChildren(h("div", { class: "scroll", style: "max-height:260px" }, h("table", {}, h("thead", {}, h("tr", {}, ["tensor", "value", "ppl", "KL", "top-1", "ok"].map(x => h("th", {}, x)))), h("tbody", {}, job.rows.map(r => h("tr", {}, h("td", { class: "mono" }, r.tensors), h("td", {}, fnum(r.value)), h("td", {}, fnum(r.perplexity, 3)), h("td", {}, fnum(r.kl_mean)), h("td", {}, r.top1_agree == null ? "—" : fnum(r.top1_agree * 100, 1) + "%"), h("td", { class: r.pass ? "ok" : "bad" }, r.pass ? "✔" : "✘"))))))); }
  const run = h("button", { class: "btn primary", disabled: ro(), onclick: async e => { e.target.disabled = true; prog.style.display = "block"; try { const { id } = await api("/api/sweep", { model: S.model, pattern: pat.value, param: "multiplier", values: vals.value.split(/[,\s]+/).filter(Boolean).map(Number), per_layer: pl.checked });
    for (;;) { const j = await api("/api/job?id=" + id); last = j; prog.firstChild.style.width = (j.total ? 100 * j.done / j.total : 0) + "%"; render(j); if (j.status !== "running") { if (j.status === "error") toast(j.error, true); break; } await new Promise(r => setTimeout(r, 600)); } } catch (er) { toast(er.message, true); } e.target.disabled = false; setTimeout(() => prog.style.display = "none", 800); } }, "Run sweep");
  root.append(h("div", { class: "card" }, h("h2", {}, "Evaluate", h("small", {}, "numpy reference forward pass")), h("div", { class: "row", style: "margin-bottom:10px" }, evalBtn("Set baseline", true), evalBtn("Evaluate now", false)), out),
    h("div", { class: "card" }, h("h2", {}, "Scale sweep", h("small", {}, "patch → eval → revert, on a working copy")), h("div", { class: "row", style: "margin-bottom:12px" }, h("span", { class: "muted" }, "tensors"), pat, h("span", { class: "muted" }, "× multipliers"), vals,
      h("label", { class: "row", style: "gap:6px" }, pl, "per layer"), run, prog), h("div", { class: "row", style: "margin-bottom:8px" }, h("span", { class: "muted" }, "metric"), msel), chart, cap, h("div", { style: "height:10px" }), table));
};

/* ───────────── shell ───────────── */
async function switchTab(t) {
  S.tab = t; const page = $("#page"); page.replaceChildren(); $("#title").textContent = TABS.find(x => x[0] === t)[1];
  $("#nav").querySelectorAll("button").forEach(b => b.setAttribute("aria-selected", b.dataset.t === t));
  if (!S.model) { page.append(h("div", { class: "empty" }, "No .gguf models found in the server root.")); return; }
  try { await refreshSummary(); await VIEWS[t](page); } catch (e) { page.replaceChildren(h("div", { class: "empty bad" }, e.message)); }
}
(async function boot() {
  const nav = $("#nav"); TABS.forEach(([id, label, d]) => nav.append(h("button", { role: "tab", "data-t": id, onclick: () => switchTab(id) }, svgIcon(d), label)));
  try { S.info = await api("/api/info"); } catch (e) { $("#page").append(h("div", { class: "empty bad" }, e.message)); return; }
  $("#mode").textContent = S.info.readonly ? "read-only" : "read / write"; $("#mode").classList.add(S.info.readonly ? "ro" : "rw"); $("#foot").textContent = "v" + S.info.version + " · " + S.info.root;
  const models = await api("/api/models"), sel = $("#model"); models.forEach(m => sel.append(h("option", { value: m.path }, m.path + "  ·  " + fmtB(m.size))));
  S.model = models[0]?.path || null; sel.addEventListener("change", () => { S.model = sel.value; S.tensor = null; S.hex = { offset: 0, sel: 0 }; switchTab(S.tab); });
  switchTab(new URLSearchParams(location.search).get("tab") || "overview");
  addEventListener("resize", () => { clearTimeout(S._rs); S._rs = setTimeout(() => switchTab(S.tab), 250); });
})();
