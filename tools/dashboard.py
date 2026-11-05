# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Training dashboard: one HTML page of SerenaBot's learning trajectories, from the result TSVs.

Sections: island evolution (results/islands.tsv, fitness per generation, one line per island),
network training (results/network-training.tsv, loss and accuracy per epoch, runs to compare) and
revisions (results/gates.tsv, params/serena/revisions.tsv). The page is self-contained: data inline,
charts drawn by its own script, hover readouts, light and dark.

Usage: uv run tools/dashboard.py [--watch seconds] [--out arena/out/training.html]
With --watch it rewrites the page every so many seconds and the page reloads itself, to follow a run.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "arena"


# ------------------------------------------------------------------ reading

def read_tsv(path):
    """(rows as dicts, comment lines) of a TSV; ([], []) when it does not exist."""
    if not path.exists():
        return [], []
    rows, comments, header = [], [], None
    for line in path.read_text().splitlines():
        if not line:
            continue
        if line.startswith("#"):
            comments.append((len(rows), line[1:].strip()))
        elif header is None:
            header = line.split("\t")
        else:
            rows.append(dict(zip(header, line.split("\t"))))
    return rows, comments


def number(text):
    try:
        value = float(text)
        return None if value != value else value  # nan -> None
    except (TypeError, ValueError):
        return None


def islands():
    """Island runs: per run its recipe, migration interval and per island the generation rows."""
    rows, comments = read_tsv(ARENA / "results/islands.tsv")
    recipes = {}
    for _, text in comments:
        match = re.match(r"(\S+) Tune (.*?): (.*)", text)
        if match:
            recipes[match.group(1)] = match.group(3)
    runs = {}
    for row in rows:
        run = runs.setdefault(row["run"], {"run": row["run"], "recipe": recipes.get(row["run"], ""), "islands": {}})
        run["islands"].setdefault(row["island"], []).append({
            "x": int(row["generation"]), "best": number(row["best_fitness"]), "mean": number(row["mean_fitness"]),
            "score": number(row["best_score"]), "bits": int(row["best_table_bits"]), "seconds": number(row["seconds"])})
    for run in runs.values():
        match = re.search(r"migrate every (\d+)", run["recipe"])
        run["migrate"] = int(match.group(1)) if match else 0
        run["islands"] = [{"label": f"island {key}", "points": points} for key, points in sorted(run["islands"].items(), key=lambda item: int(item[0]))]
    return list(runs.values())


def network():
    """Network runs: per run its command, held-out summary and epoch rows (float and quantised)."""
    rows, comments = read_tsv(ARENA / "results/network-training.tsv")
    commands, summaries, last = {}, {}, None
    for _, text in comments:
        match = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d): (.*)", text)
        if match:
            last = match.group(1)
            commands[last] = match.group(2)
        elif text.startswith("held-out") and last:
            summaries[last] = text
    runs = {}
    for row in rows:
        stamp = row["run"].split(" ")[0]
        run = runs.setdefault(row["run"], {"run": row["run"], "command": commands.get(stamp, ""),
                                          "summary": summaries.get(stamp, ""), "epochs": [], "quantised": []})
        point = {"x": int(row["epoch"]), "train": number(row["train_loss"]), "loss": number(row["validation_loss"]),
                 "accuracy": number(row["validation_accuracy"])}
        (run["quantised"] if row.get("precision") == "int" else run["epochs"]).append(point)
    return list(runs.values())


def revisions():
    gates, _ = read_tsv(ARENA / "results/gates.tsv")
    revisions, _ = read_tsv(ARENA / "params/serena/revisions.tsv")
    return {"gates": gates, "revisions": revisions}


# ------------------------------------------------------------------ the page

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
__REFRESH__<title>Serena Training</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #e2e1dc;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8984;
  --grid: #e9e8e4; --axis: #b9b8b2; --context: #c9c8c2;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #6250d6; --series-8: #e34948;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --border: #2e2e2c;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e88;
  --grid: #292927; --axis: #4a4a46; --context: #3d3d3a;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
} }
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --border: #2e2e2c;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e88;
  --grid: #292927; --axis: #4a4a46; --context: #3d3d3a;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0; }
.sub { color: var(--text-secondary); margin: 0 0 20px; }
section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 20px; }
.head { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; margin-bottom: 4px; }
.note { color: var(--text-secondary); font-size: 12.5px; margin: 4px 0 12px; overflow-wrap: anywhere; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 16px; }
.chart h3 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart text { fill: var(--text-secondary); font-size: 11px; }
.chart .label { fill: var(--text-primary); font-size: 11px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--text-secondary); margin: 2px 0 6px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 2px; border-radius: 1px; display: inline-block; }
.legend i.dash { background: none; border-top: 2px dashed var(--text-muted); height: 0; }
select { font: inherit; color: var(--text-primary); background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; max-width: 100%; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
.chip { font: inherit; font-size: 12px; color: var(--text-secondary); background: var(--surface-1); border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px; cursor: pointer; display: inline-flex; gap: 6px; align-items: center; }
.chip[aria-pressed="true"] { color: var(--text-primary); border-color: var(--text-muted); font-weight: 600; }
.chip i { width: 12px; height: 2px; display: inline-block; background: var(--context); }
.empty { color: var(--text-muted); font-style: italic; padding: 12px 0; }
.tip { position: fixed; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); display: none; z-index: 10; min-width: 140px; }
.tip .t { color: var(--text-secondary); margin-bottom: 4px; }
.tip .r { display: flex; align-items: center; gap: 8px; }
.tip .r b { color: var(--text-primary); min-width: 46px; font-variant-numeric: tabular-nums; }
.tip .r i { width: 12px; height: 2px; display: inline-block; }
.tip .r span { color: var(--text-secondary); }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--text-secondary); font-size: 12.5px; }
table { border-collapse: collapse; font-size: 12px; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 3px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
.tablewrap { overflow-x: auto; }
</style></head>
<body><main>
<h1>Serena training</h1>
<p class="sub" id="generated"></p>

<section id="islands">
  <div class="head"><h2>Island evolution</h2><select id="island-run" aria-label="Island run"></select></div>
  <p class="note" id="island-note"></p>
  <div class="legend" id="island-legend"></div>
  <div class="grid"><div class="chart" id="island-best"></div><div class="chart" id="island-mean"></div></div>
  <details><summary>Data table</summary><div class="tablewrap" id="island-table"></div></details>
</section>

<section id="network">
  <div class="head"><h2>Network training</h2></div>
  <p class="note">Pick up to four runs to compare; the others stay as grey context. Open circles: the quantised (int) network.</p>
  <div class="chips" id="net-chips"></div>
  <p class="note" id="net-note"></p>
  <div class="legend" id="net-legend"></div>
  <div class="grid"><div class="chart" id="net-loss"></div><div class="chart" id="net-train"></div><div class="chart" id="net-accuracy"></div></div>
  <details><summary>Data table</summary><div class="tablewrap" id="net-table"></div></details>
</section>

<section id="revisions">
  <div class="head"><h2>Revisions and gates</h2></div>
  <p class="note">Gate: the candidate against the newest Tank on 15x30. Sizes per revision from ./qc.sh check.</p>
  <div class="grid"><div class="chart" id="gate-score"></div><div class="chart" id="rev-jar"></div><div class="chart" id="rev-heap"></div></div>
</section>
</main>
<div class="tip" id="tip" role="tooltip"></div>
<script>
const DATA = __DATA__;
const SLOTS = 8;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const node = document.createElementNS(NS, tag); for (const k in attrs) node.setAttribute(k, attrs[k]); if (parent) parent.appendChild(node); return node; };
const fmt = v => v == null ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3);
const tip = document.getElementById("tip");
let clipCount = 0;
if (location.hash === "#light" || location.hash === "#dark") document.documentElement.setAttribute("data-theme", location.hash.slice(1));

function niceTicks(lo, hi, count) {
  const span = hi - lo || 1, step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= count) || 10 * mag;
  const ticks = []; for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

// A line chart. series: {label, color (css var) or null for context, points [{x, y}], dots: bool}.
// opts: {title, xLabel, yDomain, refs: [{y, label}], verticals: [x], markers: [{x, y, color}], intX}
function lineChart(container, series, opts) {
  container.textContent = "";
  const h3 = document.createElement("h3"); h3.textContent = opts.title; container.appendChild(h3);
  const all = series.flatMap(s => s.points).concat(opts.markers || []).filter(p => p.y != null);
  if (!all.length) { const e = document.createElement("div"); e.className = "empty"; e.textContent = "No rows yet."; container.appendChild(e); return; }
  const W = 520, H = 260, L = 46, R = 74, T = 10, B = 34;
  let x0 = Math.min(...all.map(p => p.x)), x1 = Math.max(...all.map(p => p.x));
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  let ys = all.map(p => p.y).concat((opts.refs || []).map(r => r.y));
  let [y0, y1] = opts.yDomain || [Math.min(...ys), Math.max(...ys)];
  const pad = (y1 - y0) * 0.08 || 0.05; if (!opts.yDomain) { y0 -= pad; y1 += pad; }
  const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R), sy = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title}, container);
  for (const t of niceTicks(y0, y1, 5)) {
    el("line", {x1: L, x2: W - R, y1: sy(t), y2: sy(t), stroke: css("--grid")}, svg);
    el("text", {x: L - 6, y: sy(t) + 4, "text-anchor": "end"}, svg).textContent = fmt(t).replace(/0+$/, "").replace(/\\.$/, "");
  }
  const xt = opts.intX ? niceTicks(x0, x1, 8).filter(t => Number.isInteger(t)) : niceTicks(x0, x1, 6);
  for (const t of xt) el("text", {x: sx(t), y: H - B + 16, "text-anchor": "middle"}, svg).textContent = t;
  el("line", {x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: css("--axis")}, svg);
  el("text", {x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle"}, svg).textContent = opts.xLabel;
  for (const v of opts.verticals || []) {
    el("line", {x1: sx(v), x2: sx(v), y1: T, y2: H - B, stroke: css("--axis"), "stroke-dasharray": "2 4"}, svg);
  }
  for (const r of opts.refs || []) {
    el("line", {x1: L, x2: W - R, y1: sy(r.y), y2: sy(r.y), stroke: css("--text-muted"), "stroke-dasharray": "5 4", "stroke-width": 1.25}, svg);
    el("text", {x: W - R + 6, y: sy(r.y) + 4}, svg).textContent = r.label;
  }
  const clip = "clip" + (++clipCount);
  el("rect", {x: L, y: T - 2, width: W - L - R, height: H - T - B + 4}, el("clipPath", {id: clip}, el("defs", {}, svg)));
  const plot = el("g", {"clip-path": `url(#${clip})`}, svg);
  const draw = s => {
    const pts = s.points.filter(p => p.y != null).sort((a, b) => a.x - b.x);
    if (!pts.length) return;
    const color = s.color ? css(s.color) : css("--context");
    el("path", {d: pts.map((p, i) => (i ? "L" : "M") + sx(p.x).toFixed(1) + "," + sy(p.y).toFixed(1)).join(""),
      fill: "none", stroke: color, "stroke-width": s.color ? 2 : 1.25, "stroke-linejoin": "round", "stroke-linecap": "round"}, plot);
    if (s.color && pts.length <= 15) for (const p of pts) el("circle", {cx: sx(p.x), cy: sy(p.y), r: pts.length == 1 ? 4 : 3, fill: color, stroke: css("--surface-1"), "stroke-width": 2}, plot);
  };
  series.filter(s => !s.color).forEach(draw);
  const lit = series.filter(s => s.color);
  lit.forEach(draw);
  for (const m of opts.markers || []) el("circle", {cx: sx(m.x), cy: sy(m.y), r: 5, fill: css("--surface-1"), stroke: css(m.color), "stroke-width": 2}, svg);
  // direct labels at line ends, nudged apart
  const ends = lit.map(s => { const p = s.points.filter(q => q.y != null).sort((a, b) => a.x - b.x).pop(); return p && {s, y: sy(p.y)}; }).filter(Boolean).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
  if (ends.length <= 4) for (const e of ends) el("text", {x: W - R + 6, y: e.y + 4, class: "label"}, svg).textContent = e.s.short || e.s.label;
  // hover: crosshair snapping to the nearest x, one readout of every highlighted series
  const xsAll = [...new Set(lit.flatMap(s => s.points.map(p => p.x)))].sort((a, b) => a - b);
  if (!xsAll.length) return;
  const hair = el("line", {x1: L, x2: L, y1: T, y2: H - B, stroke: css("--text-muted"), "stroke-width": 1, visibility: "hidden"}, svg);
  const hit = el("rect", {x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent", tabindex: 0}, svg);
  const show = (clientX, clientY) => {
    const box = svg.getBoundingClientRect(), vx = (clientX - box.left) / box.width * W;
    const x = xsAll.reduce((best, v) => Math.abs(sx(v) - vx) < Math.abs(sx(best) - vx) ? v : best, xsAll[0]);
    hair.setAttribute("x1", sx(x)); hair.setAttribute("x2", sx(x)); hair.setAttribute("visibility", "visible");
    tip.textContent = "";
    const t = document.createElement("div"); t.className = "t"; t.textContent = `${opts.xLabel} ${x}`; tip.appendChild(t);
    const rows = lit.map(s => ({s, p: s.points.find(p => p.x === x)})).filter(r => r.p && r.p.y != null).sort((a, b) => b.p.y - a.p.y);
    for (const r of rows) {
      const row = document.createElement("div"); row.className = "r";
      const k = document.createElement("i"); k.style.background = css(r.s.color);
      const b = document.createElement("b"); b.textContent = fmt(r.p.y);
      const sp = document.createElement("span"); sp.textContent = r.s.label + (r.p.extra ? " · " + r.p.extra : "");
      row.append(k, b, sp); tip.appendChild(row);
    }
    tip.style.display = "block";
    const tx = Math.min(clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = tx + "px"; tip.style.top = Math.max(8, clientY - tip.offsetHeight - 12) + "px";
  };
  hit.addEventListener("pointermove", e => show(e.clientX, e.clientY));
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; hair.setAttribute("visibility", "hidden"); });
  hit.addEventListener("focus", () => { const b = svg.getBoundingClientRect(); show(b.left + b.width * 0.8, b.top + 20); });
  hit.addEventListener("blur", () => { tip.style.display = "none"; hair.setAttribute("visibility", "hidden"); });
}

function legend(container, items) {
  container.textContent = "";
  for (const item of items) {
    const span = document.createElement("span"), key = document.createElement("i");
    if (item.dash) key.className = "dash"; else key.style.background = css(item.color);
    span.append(key, document.createTextNode(item.label)); container.appendChild(span);
  }
}

function table(container, header, rows) {
  container.textContent = "";
  if (!rows.length) return;
  const t = document.createElement("table"), hr = t.insertRow();
  for (const h of header) { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); }
  for (const r of rows) { const tr = t.insertRow(); for (const v of r) tr.insertCell().textContent = typeof v === "number" ? fmt(v) : v; }
  container.appendChild(t);
}

const remember = (key, value) => { try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (e) { return null; } };

// ------------------------------------------------------------------ island evolution
function renderIslands() {
  const runs = DATA.islands, select = document.getElementById("island-run");
  if (!runs.length) { document.getElementById("island-note").textContent = "No island runs yet: ./qc.sh tune --islands N."; return; }
  if (!select.options.length) {
    runs.slice().reverse().forEach(r => { const o = document.createElement("option"); o.value = r.run; o.textContent = r.run; select.appendChild(o); });
    const kept = remember("island-run"); if (kept && runs.some(r => r.run === kept)) select.value = kept;
    select.onchange = () => { remember("island-run", select.value); renderIslands(); };
  }
  const run = runs.find(r => r.run === select.value);
  document.getElementById("island-note").textContent = run.recipe + ". Fitness = score against the untouched SerenaBot minus a table-size penalty; 0.5 = even.";
  const color = i => `--series-${i % SLOTS + 1}`;
  const series = key => run.islands.map((isl, i) => ({label: isl.label, short: isl.label.replace("island ", "i"), color: color(i),
    points: isl.points.map(p => ({x: p.x, y: p[key], extra: key === "best" ? `score ${fmt(p.score)}, table 2^${p.bits}` : ""}))}));
  const all = run.islands.flatMap(i => i.points), lo = Math.min(0.4, ...all.map(p => p.mean ?? 1)), hi = Math.max(0.6, ...all.map(p => p.best ?? 0));
  const domain = [Math.floor(lo * 20) / 20, Math.ceil(hi * 20) / 20];
  const last = Math.max(...all.map(p => p.x));
  const verticals = run.migrate ? Array.from({length: Math.floor(last / run.migrate)}, (_, k) => (k + 1) * run.migrate) : [];
  legend(document.getElementById("island-legend"), run.islands.map((isl, i) => ({label: isl.label, color: color(i)}))
    .concat([{label: "even with SerenaBot", dash: true}]).concat(verticals.length ? [{label: "dotted: migration", dash: true}] : []));
  const refs = [{y: 0.5, label: "even"}];
  lineChart(document.getElementById("island-best"), series("best"), {title: "Best fitness per island", xLabel: "generation", yDomain: domain, refs, verticals, intX: true});
  lineChart(document.getElementById("island-mean"), series("mean"), {title: "Mean fitness per island", xLabel: "generation", yDomain: domain, refs, verticals, intX: true});
  table(document.getElementById("island-table"), ["island", "generation", "best", "mean", "best score", "table bits", "seconds"],
    run.islands.flatMap(isl => isl.points.map(p => [isl.label, String(p.x), p.best, p.mean, p.score, String(p.bits), p.seconds])));
}

// ------------------------------------------------------------------ network training
let picked = [];
function renderNetwork() {
  const runs = DATA.network, chips = document.getElementById("net-chips");
  if (!runs.length) { document.getElementById("net-note").textContent = "No network runs yet: uv run qcnet."; return; }
  if (!picked.length) {
    let kept = null; try { kept = JSON.parse(remember("net-picked") || "null"); } catch (e) {}
    picked = (kept || []).filter(p => runs.some(r => r.run === p.run));
    if (!picked.length) picked = runs.slice(-2).map((r, i) => ({run: r.run, slot: i}));
  }
  const slotOf = run => (picked.find(p => p.run === run) || {}).slot;
  chips.textContent = "";
  runs.slice().reverse().forEach(r => {
    const b = document.createElement("button"), k = document.createElement("i"), slot = slotOf(r.run);
    b.className = "chip"; b.setAttribute("aria-pressed", slot != null); b.title = r.command;
    if (slot != null) k.style.background = css(`--series-${slot + 1}`);
    b.append(k, document.createTextNode(r.run));
    b.onclick = () => {
      if (slot != null) picked = picked.filter(p => p.run !== r.run);
      else if (picked.length < 4) { const used = picked.map(p => p.slot); picked.push({run: r.run, slot: [0, 1, 2, 3].find(s => !used.includes(s))}); }
      remember("net-picked", JSON.stringify(picked)); renderNetwork();
    };
    chips.appendChild(b);
  });
  const lit = runs.filter(r => slotOf(r.run) != null);
  const note = document.getElementById("net-note"); note.textContent = "";
  for (const r of lit) { const d = document.createElement("div"); d.textContent = `${r.run}: ${r.command}${r.summary ? " — " + r.summary : ""}`; note.appendChild(d); }
  legend(document.getElementById("net-legend"), lit.map(r => ({label: r.run, color: `--series-${slotOf(r.run) + 1}`})).concat([{label: "reference", dash: true}]));
  const series = key => runs.map(r => ({label: r.run, short: r.run.split(" ").slice(1).join(" ") || r.run,
    color: slotOf(r.run) != null ? `--series-${slotOf(r.run) + 1}` : null, points: r.epochs.map(p => ({x: p.x, y: p[key]}))}));
  const markers = key => lit.flatMap(r => r.quantised.filter(p => p[key] != null).map(p => ({x: p.x, y: p[key], color: `--series-${slotOf(r.run) + 1}`})));
  const litPts = key => lit.flatMap(r => r.epochs.concat(r.quantised).map(p => p[key])).filter(v => v != null);
  const domain = (key, refs) => { const v = litPts(key).concat(refs); if (!v.length) return null; const lo = Math.min(...v), hi = Math.max(...v), pad = (hi - lo) * 0.1 || 0.02; return [lo - pad, hi + pad]; };
  lineChart(document.getElementById("net-loss"), series("loss"), {title: "Held-out loss", xLabel: "epoch", refs: [{y: 0.6931, label: "coin flip"}], markers: markers("loss"), yDomain: domain("loss", [0.6931]), intX: true});
  lineChart(document.getElementById("net-train"), series("train"), {title: "Training loss", xLabel: "epoch", yDomain: domain("train", []), intX: true});
  lineChart(document.getElementById("net-accuracy"), series("accuracy"), {title: "Held-out accuracy, decisive games", xLabel: "epoch", refs: [{y: 0.5, label: "chance"}], markers: markers("accuracy"), yDomain: domain("accuracy", [0.5]), intX: true});
  table(document.getElementById("net-table"), ["run", "epoch", "train loss", "held-out loss", "accuracy", "precision"],
    lit.flatMap(r => r.epochs.map(p => [r.run, String(p.x), p.train, p.loss, p.accuracy, "float"]).concat(r.quantised.map(p => [r.run, String(p.x), p.train, p.loss, p.accuracy, "int"]))));
}

// ------------------------------------------------------------------ revisions and gates
function renderRevisions() {
  const g = DATA.revisions.gates, r = DATA.revisions.revisions;
  const one = (label, rows, key, xKey) => [{label, color: "--series-1", points: rows.map((row, i) => ({x: i + 1, y: parseFloat(row[key]), extra: row[xKey]})).filter(p => !isNaN(p.y))}];
  lineChart(document.getElementById("gate-score"), one("gate score", g, "score", "candidate"), {title: "Gate score", xLabel: "gate", refs: [{y: 0.5, label: "even"}], yDomain: [0, 1], intX: true});
  lineChart(document.getElementById("rev-jar"), one("upload JAR KB", r, "jar_kb", "name"), {title: "Upload JAR (KB)", xLabel: "revision", intX: true});
  lineChart(document.getElementById("rev-heap"), one("heap MB", r, "heap_mb", "name"), {title: "Heap (MB)", xLabel: "revision", refs: [{y: 80, label: "80 MB cap"}], intX: true});
}

function renderAll() { renderIslands(); renderNetwork(); renderRevisions(); }
document.getElementById("generated").textContent = `Generated ${DATA.generated} from arena/results and arena/params/serena.` + (DATA.watch ? ` Reloads every ${DATA.watch} s.` : " Rerun ./qc.sh dashboard to refresh.");
renderAll();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
</script>
</body></html>
"""


def write(out, watch):
    data = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "watch": watch,
            "islands": islands(), "network": network(), "revisions": revisions()}
    blob = json.dumps(data).replace("</", "<\\/")
    refresh = f'<meta http-equiv="refresh" content="{watch}">\n' if watch else ""
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(PAGE.replace("__REFRESH__", refresh).replace("__DATA__", blob))
    temporary.replace(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", type=int, default=0, help="rewrite every so many seconds; the page reloads itself")
    parser.add_argument("--out", type=Path, default=ARENA / "out/training.html")
    args = parser.parse_args()
    write(args.out, args.watch)
    print(f"wrote {args.out}")
    while args.watch:
        time.sleep(args.watch)
        write(args.out, args.watch)


if __name__ == "__main__":
    main()
