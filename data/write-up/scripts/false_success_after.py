"""After the first false-success reply, is new code submitted? 7B no-exit, alpha 0.5. Run from repo root."""
import json, re, pandas as pd
from collections import defaultdict
L = pd.read_parquet("data/write-up/labels/noexit/labels.parquet")
key = json.load(open("data/write-up/labels/noexit/key.json"))
L = L[(L["size"]=="7b") & ((L.a==0.5)|(L.a==0.0))]
cache = {}
def ep(eid):
    k = key[eid]
    if k["file"] not in cache:
        cache[k["file"]] = [json.loads(l) for l in open(k["file"])]
    for r in cache[k["file"]]:
        if r["episode"]==k["episode"] and r["vector"]==k["vector"] and r["alpha"]==k["alpha"]: return r
def code(t):
    m = re.findall(r"```(?:python)?\n(.*?)```", t, re.S)
    return re.sub(r"\s+"," ", m[-1]).strip() if m else None
rows=[]
for eid, g in L.groupby("eid"):
    r = ep(eid); asst=[m["content"] for m in r["messages"] if m["role"]=="assistant"]
    codes=[code(t) for t in asst]
    fs = g[g.claims_success==1].reply.min()
    # "anchor": first false-success reply; for episodes without one, use reply 3 as a comparison point
    anchor = int(fs) if pd.notna(fs) else None
    after = range((anchor or 3), len(asst))  # replies after anchor (0-based index = reply number)
    new=rep=none=0
    for i in after:
        c=codes[i]
        if c is None: none+=1
        elif c in codes[:i]: rep+=1
        else: new+=1
    rows.append(dict(eid=eid,cond=g.cond.iloc[0],grp=g.grp.iloc[0],fs=anchor is not None,n_after=len(after),new=new,rep=rep,none=none,
                     fs_replies=int(g.claims_success.sum()), n=len(asst)))
D=pd.DataFrame(rows)
pos={"+calm","+excited","−afraid","−angry","−ashamed","−desperate","−disgusted","−lonely"}
D["set"]=D.cond.map(lambda c:"positive-valence" if c in pos else ("unsteered/random" if c in {"none","+random","−random"} else "other"))
D=D[D.set!="other"]
s=D.groupby(["set","fs"])[["n_after","new","rep","none"]].sum()
s["new%"]=(100*s.new/s.n_after).round(0); s["repeat%"]=(100*s.rep/s.n_after).round(0); s["nocode%"]=(100*s.none/s.n_after).round(0)
s["episodes"]=D.groupby(["set","fs"]).size()
print(s)
# episodes that ever produce new code after first false-success
F=D[D.fs]
print("\nFS episodes:",len(F)," with zero new code after FS:",(F.new==0).sum(), " mean FS replies per FS episode:",F.fs_replies.mean().round(1))
print(F.groupby("set").apply(lambda x: pd.Series({"eps":len(x),"zero_new_after":(x.new==0).sum(),"mean_fs_replies":x.fs_replies.mean().round(1)})))
print(F.groupby("set").apply(lambda x: pd.Series({"eps":len(x),"zero_new_after":(x.new==0).sum(),"mean_fs_replies":x.fs_replies.mean().round(1)})))
# examples: positive-valence FS episodes with no new code after the claim
print(D[(D.set=="positive-valence")&D.fs&(D.new==0)].sort_values("fs_replies",ascending=False).head(4)[["eid","cond","fs_replies","n"]].to_string(index=False))
print(D[(D.set=="unsteered/random")&D.fs][["eid","cond","new","n"]].to_string(index=False))
