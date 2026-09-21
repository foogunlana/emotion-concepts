"""Run from the repo root: uv run python data/write-up/scripts/all_emotions_to_inspect.py

Every episode behind the "All emotions at 7B" table (7B, impossible Fast Sum / noexit, α = 0.5): one log per
row, 32 episodes each, unselected, with hand labels on every attempt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from valence_to_inspect import EPS, W, write

OUT = W / "7B-all-emotions"; OUT.mkdir(exist_ok=True)
for f in OUT.glob("*.eval"): f.unlink()
ROWS = [("positive", ["−afraid", "−angry", "−ashamed", "+calm", "−desperate", "−disgusted", "+excited", "−lonely"]),
        ("negative", ["+afraid", "+angry", "+ashamed", "−calm", "+desperate", "−excited", "−joyful"]),
        ("baseline", ["none", "+random", "−random"])]
n = 0
for group, conds in ROWS:
    for c in conds:
        n += 1
        e = [x for x in EPS if x["size"] == "7b" and x["arm"] == "noexit" and x["cond"] == c and x["a"] in (0.0, 0.5)]
        assert len(e) == 32, (c, len(e))
        name = "unsteered" if c == "none" else c
        slug = name.replace("+", "plus_").replace("−", "minus_")
        write(f"{n:02d}_{group}_{slug}.eval", f"{n:02d} · 7B {name} (α=0.5) · {group} · impossible task · all 32",
              e, f"All 32 episodes of {name}, unselected. Row of the 'All emotions at 7B' table.", out=OUT)
