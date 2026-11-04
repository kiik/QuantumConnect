# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Draws the charts of docs/methodology.md from the result TSVs, as SVG readable in light and dark.

Each TSV is read from arena/ when it exists there, otherwise from the first given git ref that has it
(git show <ref>:<path>), so charts can be drawn from branches that are not merged yet.

Usage: uv run tools/figures.py [git-ref ...]
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
WIDTH, HEIGHT, LEFT, RIGHT, TOP, BOTTOM = 640, 340, 60, 170, 40, 50
STYLE = """<style>
  .text { fill: #374151; font: 12px sans-serif; }
  .axis { stroke: #9ca3af; stroke-width: 1; }
  .grid { stroke: #9ca3af; stroke-opacity: .25; }
  .muted { stroke: #9ca3af; }
  @media (prefers-color-scheme: dark) { .text { fill: #d1d5db; } }
</style>"""
COLOURS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]


# ------------------------------------------------------------------ reading

def read_tsv(path, refs):
    """Rows of a TSV as dicts, without its # lines; None when no source has it."""
    local = ROOT / path
    if local.exists():
        text = local.read_text()
    else:
        text = None
        for ref in refs:
            shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{path}"],
                                   capture_output=True, text=True)
            if shown.returncode == 0:
                text = shown.stdout
                break
        if text is None:
            return None
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


# ------------------------------------------------------------------ drawing

def line_chart(title, x_label, y_label, series, reference=None):
    """series: (label, colour or None for muted, [(x, y)]); reference: (label, y) drawn dashed."""
    xs = [x for _, _, points in series for x, _ in points]
    ys = [y for _, _, points in series for _, y in points] + ([reference[1]] if reference else [])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.05 or 0.05
    y0, y1 = y0 - pad, y1 + pad
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def sx(x):
        return LEFT + (x - x0) / ((x1 - x0) or 1) * plot_w

    def sy(y):
        return TOP + (y1 - y) / (y1 - y0) * plot_h

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}">', STYLE,
           f'<text class="text" x="{LEFT}" y="22" font-weight="bold">{title}</text>']
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        out.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{sy(y):.1f}" y2="{sy(y):.1f}"/>')
        out.append(f'<text class="text" x="{LEFT - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.2f}</text>')
    for x in sorted(set(xs)):
        if len(set(xs)) <= 12 or x % 5 == 0:
            out.append(f'<text class="text" x="{sx(x):.1f}" y="{TOP + plot_h + 16}" text-anchor="middle">{x:g}</text>')
    out.append(f'<line class="axis" x1="{LEFT}" x2="{LEFT + plot_w}" y1="{TOP + plot_h}" y2="{TOP + plot_h}"/>')
    out.append(f'<text class="text" x="{LEFT + plot_w / 2}" y="{HEIGHT - 10}" text-anchor="middle">{x_label}</text>')
    out.append(f'<text class="text" transform="translate(14 {TOP + plot_h / 2}) rotate(-90)" '
               f'text-anchor="middle">{y_label}</text>')
    if reference:
        out.append(f'<line class="muted" stroke-dasharray="4 4" x1="{LEFT}" x2="{LEFT + plot_w}" '
                   f'y1="{sy(reference[1]):.1f}" y2="{sy(reference[1]):.1f}"/>')
    legend_y = TOP + 6
    for label, colour, points in sorted(series, key=lambda s: s[1] is not None):
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        if colour:
            out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.5" points="{path}"/>')
            out += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{colour}"/>' for x, y in points]
        else:
            out.append(f'<polyline class="muted" fill="none" stroke-width="1.2" points="{path}"/>')
        if label:
            stroke = f'stroke="{colour}"' if colour else 'class="muted"'
            out.append(f'<line {stroke} stroke-width="2.5" x1="{WIDTH - RIGHT + 12}" x2="{WIDTH - RIGHT + 28}" '
                       f'y1="{legend_y}" y2="{legend_y}"/>')
            out.append(f'<text class="text" x="{WIDTH - RIGHT + 34}" y="{legend_y + 4}">{label}</text>')
            legend_y += 20
    if reference:
        out.append(f'<line class="muted" stroke-dasharray="4 4" x1="{WIDTH - RIGHT + 12}" x2="{WIDTH - RIGHT + 28}" '
                   f'y1="{legend_y}" y2="{legend_y}"/>')
        out.append(f'<text class="text" x="{WIDTH - RIGHT + 34}" y="{legend_y + 4}">{reference[0]}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ the charts

def network_training(rows):
    """Held-out loss per epoch of every float run: the first run and the final run highlighted."""
    runs = {}
    for row in rows:
        if row["precision"] == "float":
            runs.setdefault(row["run"], []).append((int(row["epoch"]), float(row["validation_loss"])))
    names = list(runs)
    series = [("tuning runs" if i == 0 else None, None, runs[name]) for i, name in enumerate(names[1:-1])]
    series.append(("first run", COLOURS[1], runs[names[0]]))
    series.append(("final run", COLOURS[0], runs[names[-1]]))
    return line_chart("Network: held-out loss per epoch", "epoch", "cross-entropy", series,
                      ("coin flip 0.693", 0.6931))


def islands(rows):
    """Best fitness per generation, one line per island of the newest run."""
    newest = max(row["run"] for row in rows)
    by_island = {}
    for row in rows:
        if row["run"] == newest:
            by_island.setdefault(row["island"], []).append((int(row["generation"]), float(row["best_fitness"])))
    series = [(f"island {island}", COLOURS[i % len(COLOURS)], sorted(points))
              for i, (island, points) in enumerate(sorted(by_island.items()))]
    return line_chart(f"Islands: best fitness per generation ({newest})", "generation", "fitness", series)


def main(refs):
    FIGURES.mkdir(parents=True, exist_ok=True)
    charts = [("arena/results/network-training.tsv", "network-training.svg", network_training),
              ("arena/results/islands.tsv", "islands-fitness.svg", islands)]
    for source, name, draw in charts:
        rows = read_tsv(source, refs)
        if not rows:
            print(f"{source}: no rows, {name} not drawn")
            continue
        (FIGURES / name).write_text(draw(rows))
        print(f"{name}: {len(rows)} rows from {source}")
    for source in ["arena/results/gates.tsv", "arena/params/serena/revisions.tsv"]:
        rows = read_tsv(source, refs)
        print(f"{source}: {'missing' if rows is None else f'{len(rows)} rows'} (shown as a table, not charted)")


if __name__ == "__main__":
    main(sys.argv[1:])
