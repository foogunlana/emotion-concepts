"""No-exit false success and 'not possible' for ±calm, ±desperate and unsteered at every model size. Run from repo root."""
import pandas as pd
E = pd.read_parquet("data/write-up/labels/noexit/episodes_final.parquet")
print(E.groupby(["size", "a"]).cond.nunique().rename("conditions").to_string(), "\n")
s = E[E.cond.isin(["+calm", "−calm", "none", "+desperate", "−desperate"])]
t = s.groupby(["size", "a", "cond"])[["succ", "imp"]].mean().mul(100).round(0)
t["n"] = s.groupby(["size", "a", "cond"]).size()
print(t.rename(columns={"succ": "false success %", "imp": "not possible %"}).to_string())
