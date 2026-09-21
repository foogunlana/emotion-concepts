"""Run from the repo root:
    uv run python data/write-up/scripts/table_calm_to_inspect.py                      # +calm  -> 7B-calm-positive
    uv run python data/write-up/scripts/table_calm_to_inspect.py −desperate 7B-desperate-negative

Every episode behind a write-up table (7B, impossible Fast Sum / noexit, α = 0.5): the steered condition,
unsteered, +random and −random, 32 each — 128 in all, unselected, with hand labels on every attempt.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from valence_to_inspect import EPS, W, write

cond, folder = (sys.argv[1].replace("-", "−", 1) if sys.argv[1].startswith("-") else sys.argv[1], sys.argv[2]) \
    if len(sys.argv) > 2 else ("+calm", "7B-calm-positive")
OUT = W / folder; OUT.mkdir(exist_ok=True)
for f in OUT.glob("*.eval"): f.unlink()
def eps(c):
    return [e for e in EPS if e["size"] == "7b" and e["arm"] == "noexit" and e["cond"] == c and e["a"] in (0.0, 0.5)]
slug = cond.replace("+", "plus_").replace("−", "minus_")
for fname, c, title in [
    (f"1_{slug}.eval", cond, f"Table · 7B {cond} (α=0.5) · impossible task · all 32"),
    ("2_unsteered.eval", "none", "Table · 7B unsteered · impossible task · all 32"),
    ("3_random_plus.eval", "+random", "Table · 7B +random (α=0.5) · impossible task · all 32"),
    ("4_random_minus.eval", "−random", "Table · 7B −random (α=0.5) · impossible task · all 32"),
]:
    e = eps(c); assert len(e) == 32, (c, len(e))
    write(fname, title, e, f"All 32 episodes of {c}, unselected. Row of the write-up table.", out=OUT)
