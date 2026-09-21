"""Solvable arm: merge blinded labels, then (1) main conditions vs random, (2) desperate dose curve (calibration)."""
import json, glob, re
from pathlib import Path
import polars as pl
from scipy.stats import fisher_exact
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(230)
L = Path("/private/tmp/claude-502/-Users-bo-code-oss-emotion-concepts/468611fa-7732-4193-a003-afd77657a76d/scratchpad/labels_solv")
key = json.load(open(L / "key.json"))
lab = {(x["id"], int(x["reply"])): x for f in glob.glob(str(L / "out/batch_*.jsonl")) for x in map(json.loads, filter(str.strip, open(f)))}
dup = {}
for f in glob.glob(str(L / "batches/*.txt")):
    for m in re.finditer(r"^--- (S\d{4}) reply (\d+): IDENTICAL to reply (\d+)", open(f).read(), re.M):
        dup[(m[1], int(m[2]))] = (m[1], int(m[3]))
VAL = {"afraid":-1,"angry":-1,"ashamed":-1,"desperate":-1,"disgusted":-1,"lonely":-1,"sad":-1,"calm":1,"excited":1,"joyful":1,"proud":1,"surprised":0}
rows, missing = [], 0
for eid, k in key.items():
    r = json.loads(open(k["file"]).readlines()[k["episode"]]); m = r["messages"]
    asst = [x for x in m if x["role"] == "assistant"]
    ls = []
    for i in range(len(asst)):
        x = lab.get(dup.get((eid, i + 1), (eid, i + 1)))
        if x is None: missing += 1; continue
        fb = m[2 * i + 2]["content"] if 2 * i + 2 < len(m) else ""
        ls.append(x | {"after_fail": i > 0})
    if not ls: continue
    c = "none" if k["vector"] == "none" else ("+" if k["alpha"] > 0 else "−") + k["vector"]
    v = k["vector"]; sign = 1 if k["alpha"] > 0 else -1
    grp = "unsteered" if c == "none" else "random" if v == "random" else "control" if v in ("shuffled", "neutral") else \
          ("positive" if sign * VAL[v] > 0 else "negative" if sign * VAL[v] < 0 else "neutral-emotion")
    rows.append(dict(eid=eid, step=k["step"], size=k["size"], a=abs(k["alpha"]), alpha=k["alpha"], cond=c, grp=grp,
                     solved=r["outcome"] == "solved", attempts=len(asst),
                     imp=any(x["claim"] == "impossible" for x in ls), succ=any(x["claims_success"] for x in ls),
                     distress=any(x["distress"] for x in ls), gaveup=any(x["gives_up"] for x in ls), offtask=any(x["off_task"] for x in ls)))
E = pl.DataFrame(rows); E.write_parquet(L / "episodes.parquet")
print(f"episodes {E.height}, missing reply labels {missing}")
M = ["solved", "imp", "succ", "distress", "gaveup", "offtask"]
USED = {"1.5b": 0.3, "3b": 0.5, "7b": 0.5, "14b": 0.5}
print("\n## Main solvable arm, α used per size — episodes with ≥1 labelled reply (* p<.05, ** p<.01 vs pooled ±random)")
for size in ["1.5b", "3b", "7b", "14b"]:
    x = E.filter(pl.col("step") == "main", pl.col("size") == size, (pl.col("cond") == "none") | (pl.col("a") == USED[size]))
    rnd = x.filter(pl.col("grp") == "random")
    out = []
    for c in ["none", "±random"] + sorted(set(x["cond"]) - {"none", "+random", "−random"}):
        y = x.filter(pl.col("cond") == "none") if c == "none" else rnd if c == "±random" else x.filter(pl.col("cond") == c)
        if y.height == 0: continue
        row = {"cond": c, "grp": y["grp"][0], "n": y.height}
        for mm in M:
            k_ = int(y[mm].sum()); s = str(k_)
            if c not in ("none", "±random") and rnd.height:
                kr = int(rnd[mm].sum()); p = fisher_exact([[k_, y.height - k_], [kr, rnd.height - kr]])[1]
                s += " **" if p < .01 else " *" if p < .05 else ""
            row[mm] = s
        out.append(row)
    print(f"\n### {size}"); print(pl.DataFrame(out))
print("\n## Dose curve: calibration, desperate at each α (n=16 each) — solvable")
cal = E.filter(pl.col("step") == "calib")
print(cal.group_by("size", "alpha").agg(pl.len().alias("n"), *[pl.col(mm).sum() for mm in M]).sort("size", "alpha"))
