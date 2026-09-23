"""Exit variant (the model may reply INFEASIBLE): how often and how early each 7B condition gives up. Run from repo root."""
import json, glob, re, pandas as pd
rows = []
for f in glob.glob("data/behaviours-merged/qwen2.5-coder-7b-instruct/episodes/main_exit_*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["alpha"] not in (0.5, -0.5, 0.0): continue
        s = "+" if r["alpha"] > 0 else ("−" if r["alpha"] < 0 else "")
        rows.append(dict(cond=s + r["vector"] if r["alpha"] else "unsteered", outcome=r["outcome"],
                         attempts=r["attempts"], conceded_at=r.get("conceded_at")))
D = pd.DataFrame(rows)
t = pd.crosstab(D.cond, D.outcome, normalize="index").mul(100).round(0)
t["n"] = D.groupby("cond").size()
t["median attempt of INFEASIBLE"] = D[D.outcome == "gave_up"].groupby("cond").conceded_at.median()
print(t.sort_values("gave_up" if "gave_up" in t else t.columns[0]).to_string())
