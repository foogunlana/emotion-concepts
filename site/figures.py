"""Figures for the write-up site, drawn as inline SVG from the labelled data.

Each figure is a function returning an SVG string. Colours are CSS variables from site/style.css, so the
chart follows the page's light/dark theme; every bar carries a <title> tooltip. The numbers are computed
here from data/write-up/labels/noexit/episodes_final.parquet, not copied from the tables.
"""
from html import escape
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "data" / "write-up" / "labels" / "noexit" / "episodes_final.parquet"

GROUPS = [
    ("Positive valence", ["−afraid", "−angry", "−ashamed", "+calm", "−desperate", "−disgusted", "+excited", "−lonely"]),
    ("Negative valence", ["+afraid", "+angry", "+ashamed", "−calm", "+desperate", "−excited", "−joyful"]),
    ("Baselines", ["none", "+random", "−random"]),
]


def _rates():
    ep = pl.read_parquet(EPISODES).filter(pl.col("size") == "7b", pl.col("a").is_in([0.0, 0.5]))
    t = ep.group_by("cond").agg(pl.len().alias("n"), pl.col("succ").sum().alias("succ"), pl.col("imp").sum().alias("imp"))
    return {r["cond"]: r for r in t.iter_rows(named=True)}


def _bar(x0, y, w, h, side, r=4):
    """A horizontal bar anchored at x0, rounded only at its outer end. side = +1 (right) or -1 (left)."""
    if w <= 0:
        return ""
    r = min(r, h / 2, w)
    if side > 0:
        return (f"M{x0:.1f},{y:.1f}H{x0 + w - r:.1f}Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f}"
                f"V{y + h - r:.1f}Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f}H{x0:.1f}Z")
    return (f"M{x0:.1f},{y:.1f}H{x0 - w + r:.1f}Q{x0 - w:.1f},{y:.1f} {x0 - w:.1f},{y + r:.1f}"
            f"V{y + h - r:.1f}Q{x0 - w:.1f},{y + h:.1f} {x0 - w + r:.1f},{y + h:.1f}H{x0:.1f}Z")


def valence_butterfly() -> str:
    """Per steering condition at 7B: 'not possible' to the left, 'false success' to the right."""
    rates = _rates()
    W, label_w, pad, arm = 700, 104, 40, 248
    xc = label_w + pad + arm                       # centre line
    row, bar_h, top = 22, 12, 56
    n_rows = sum(len(c) + 1 for _, c in GROUPS)
    H = top + n_rows * row + 34
    txt = 'font-family="Inter, -apple-system, sans-serif"'
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="vb-title" xmlns="http://www.w3.org/2000/svg">',
           '<title id="vb-title">Share of episodes with a false success claim (right) or a "not possible" claim (left), '
           'per steering condition, Qwen2.5-Coder-7B at α = 0.5</title>']
    # legend, directly over each arm
    out.append(f'<g {txt} font-size="12.5" fill="var(--fg-2)">'
               f'<rect x="{xc - arm}" y="10" width="12" height="12" rx="3" fill="var(--viz-red)"/>'
               f'<text x="{xc - arm + 18}" y="20.5">says the task is not possible</text>'
               f'<rect x="{xc + 12}" y="10" width="12" height="12" rx="3" fill="var(--viz-blue)"/>'
               f'<text x="{xc + 30}" y="20.5">claims failed code works</text></g>')
    grid_y0, grid_y1 = top - 8, top + n_rows * row
    # gridlines and baseline
    for f in (0.25, 0.5, 0.75, 1.0):
        for s in (-1, 1):
            x = xc + s * f * arm
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{grid_y0}" y2="{grid_y1}" stroke="var(--rule)" stroke-width="1"/>')
    out.append(f'<line x1="{xc}" x2="{xc}" y1="{grid_y0}" y2="{grid_y1}" stroke="var(--axis)" stroke-width="1.5"/>')
    # axis labels
    out.append(f'<g {txt} font-size="11" fill="var(--muted)" text-anchor="middle">')
    for f, s in ((1.0, -1), (0.5, -1), (0.0, 1), (0.5, 1), (1.0, 1)):
        out.append(f'<text x="{xc + s * f * arm:.1f}" y="{grid_y1 + 18}">{int(f * 100)}%</text>')
    out.append(f'<text x="{xc}" y="{grid_y1 + 32}" font-size="11">share of 32 episodes</text></g>')
    y = top
    for g, conds in GROUPS:
        out.append(f'<text {txt} x="0" y="{y + 11}" font-size="12" font-weight="600" fill="var(--fg-2)">{escape(g)}</text>')
        y += row
        for c in conds:
            r = rates[c]
            n, s, i = r["n"], r["succ"], r["imp"]
            name = "unsteered" if c == "none" else c
            ws, wi = s / n * arm, i / n * arm
            by = y + (row - bar_h) / 2 - 3
            tip = f"{name}: false success {s}/{n} ({s / n:.0%}), not possible {i}/{n} ({i / n:.0%})"
            out.append(f'<g><title>{escape(tip)}</title>'
                       f'<rect x="0" y="{y - 3}" width="{W}" height="{row}" fill="transparent"/>'
                       f'<text {txt} x="{label_w}" y="{y + 10}" font-size="12.5" text-anchor="end" fill="var(--fg)">{escape(name)}</text>')
            if wi:
                out.append(f'<path d="{_bar(xc - 1, by, wi, bar_h, -1)}" fill="var(--viz-red)"/>'
                           f'<text {txt} x="{xc - wi - 5:.1f}" y="{y + 10}" font-size="11" text-anchor="end" '
                           f'fill="var(--fg-2)">{i / n:.0%}</text>')
            if ws:
                out.append(f'<path d="{_bar(xc + 1, by, ws, bar_h, 1)}" fill="var(--viz-blue)"/>'
                           f'<text {txt} x="{xc + ws + 5:.1f}" y="{y + 10}" font-size="11" fill="var(--fg-2)">{s / n:.0%}</text>')
            out.append('</g>')
            y += row
    out.append('</svg>')
    return "\n".join(out)


FIGURES = {"valence": valence_butterfly}


LIGHT = {"var(--fg-2)": "#52514e", "var(--fg)": "#0b0b0b", "var(--muted)": "#898781", "var(--rule)": "#e1e0d9",
         "var(--axis)": "#c3c2b7", "var(--viz-red)": "#e34948", "var(--viz-blue)": "#2a78d6"}


def static_svg(name: str) -> str:
    """The same figure with fixed light-theme colours and a background, for places without the page CSS
    (the repo README on GitHub)."""
    svg = FIGURES[name]()
    for var, hex_ in LIGHT.items():
        svg = svg.replace(var, hex_)
    return svg.replace("<svg ", '<svg style="background:#fcfcfb" ', 1)
