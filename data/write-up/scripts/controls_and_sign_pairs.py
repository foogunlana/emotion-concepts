"""Controls, sign pairs and off-task rates for amendment 2. 7B no-exit, alpha 0.5. Run from repo root."""
import pandas as pd
E = pd.read_parquet("data/write-up/labels/noexit/episodes_final.parquet")
E = E[(E["size"] == "7b") & (E.a.isin([0.5, 0.0]))]
t = E.groupby(["grp", "cond"])[["imp", "succ", "offtask"]].mean().mul(100).round(0)
t["n"] = E.groupby(["grp", "cond"]).size()
t.columns = ["not possible %", "false success %", "off task %", "n"]
print(t.to_string(), "\n")
r = t.droplevel(0)
for e in ["calm", "desperate", "angry", "ashamed", "afraid", "excited"]:
    p, m = r.loc["+" + e], r.loc["−" + e]
    print(f"{e:10s} towards: FS {p['false success %']:.0f}% NP {p['not possible %']:.0f}%   "
          f"away: FS {m['false success %']:.0f}% NP {m['not possible %']:.0f}%")
