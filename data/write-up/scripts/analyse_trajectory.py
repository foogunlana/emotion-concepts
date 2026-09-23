"""Merge the trajectory labels, unblind them, and summarise: did the model keep trying; if it stopped, why; what kind
of false success; how many genuine attempts after the first event. Also reliability on the double-labelled batches.
Run from repo root after the labelling."""
import glob, json, pandas as pd
T = "data/write-up/labels/trajectory"
key = json.load(open("data/write-up/labels/noexit/key.json"))
POS = {"+calm", "+excited", "−afraid", "−angry", "−ashamed", "−desperate", "−disgusted", "−lonely"}
def cond(k):
    if k["alpha"] == 0: return "unsteered"
    return ("+" if k["alpha"] > 0 else "−") + k["vector"]
def group(c):
    if c == "unsteered" or "random" in c: return c
    return "positive valence" if c in POS else "negative valence"
def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l); a = r["attempts"]; n = len(a)
            new = [x["n"] for x in a if x["new"]]
            ev = [(x["n"], x["event"]) for x in a if x["event"] in ("false_success", "impossible")]
            stop = max(new) if new else 0
            first_n, first_ev = ev[0] if ev else (None, "none")
            rows.append(dict(id=r["id"], cond=cond(key[r["id"]]), attempts=n, n_new=len(new), stop_attempt=stop,
                             kept_trying=stop >= n - 1, stop_reason=r["stop_reason"], stop_other=r.get("stop_other", ""),
                             fs_type=r["fs_type"], first_event=first_ev, first_event_attempt=first_n,
                             new_after_first_event=sum(1 for k in new if first_n and k > first_n) if first_n else None,
                             attempts_to_stop=(stop - first_n) if first_n and stop >= first_n else (0 if first_n else None),
                             quote=r.get("quote", ""), notes=r.get("notes", ""), batch=f.split("/")[-1][:-6]))
    D = pd.DataFrame(rows); D["group"] = D.cond.map(group); return D
if __name__ == "__main__":
    D = load(f"{T}/out/batch_*.jsonl")
    # second pass: 'other' stops re-read with the text-first rules (RUBRIC_harmonise.md); keep the first pass alongside
    D["stop_reason_first"], D["stop_other_first"] = D.stop_reason, D.stop_other
    D["stop_fs_type"] = D.fs_type.where(D.stop_reason == "false_success", "none")
    H = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(f"{T}/out_harmonise/*.jsonl") for l in open(f) if l.strip()}
    for i, r in D.iterrows():
        if r.id in H:
            h = H[r.id]
            D.loc[i, ["stop_reason", "stop_other", "stop_fs_type"]] = [h["stop_reason"], h.get("stop_other", ""), h["fs_type"]]
            D.loc[i, "quote"] = h.get("quote", r.quote)
    D["harmonised"] = D.id.isin(H)
    D.to_parquet(f"{T}/trajectory.parquet")
    if H: print(f"second pass applied to {len(H)} episodes:\n", pd.crosstab(D[D.harmonised].group, D[D.harmonised].stop_reason).to_string(), "\n")
    print(f"{len(D)} episodes labelled\n")
    order = ["positive valence", "negative valence", "unsteered", "+random", "−random"]
    g = D.groupby("group")
    s = pd.DataFrame({"episodes": g.size(), "kept trying %": g.kept_trying.mean().mul(100).round(0),
                      "median genuine attempts": g.n_new.median()}).reindex(order)
    print(s.to_string(), "\n")
    print("stop reason (% of episodes)\n", pd.crosstab(D.group, D.stop_reason, normalize="index").mul(100).round(0).reindex(order).to_string(), "\n")
    F = D[D.stop_reason == "false_success"]
    print("type of false success, among episodes that stop on it (counts)\n",
          pd.crosstab(F.group, F.stop_fs_type, margins=True).reindex(order + ["All"]).to_string(), "\n")
    print("stop 'other' tags\n", pd.crosstab(D[D.stop_reason == "other"].group, D[D.stop_reason == "other"].stop_other).to_string(), "\n")
    E = D[D.first_event != "none"]
    print("after the first false-success / impossible event: genuine new attempts afterwards, and attempts until the stop\n",
          E.groupby(["group", "first_event"]).agg(episodes=("id", "size"), median_new_after=("new_after_first_event", "median"),
          no_new_after_pct=("new_after_first_event", lambda x: round(100 * (x == 0).mean())),
          median_attempts_to_stop=("attempts_to_stop", "median")).to_string(), "\n")
    c = D.groupby("cond").agg(n=("id", "size"), kept_trying=("kept_trying", "mean"),
                              stop_fs=("stop_reason", lambda x: (x == "false_success").mean()),
                              stop_imp=("stop_reason", lambda x: (x == "impossible").mean()),
                              stop_other=("stop_reason", lambda x: (x == "other").mean()))
    print("by condition (%)\n", (c.drop(columns="n").mul(100).round(0)).assign(n=c.n).to_string(), "\n")
    # reliability
    R = load(f"{T}/out_reliability/batch_*.jsonl")
    if len(R):
        M = D.merge(R, on="id", suffixes=("", "_2"))
        def kappa(a, b):
            po = (a == b).mean(); pe = sum((a == k).mean() * (b == k).mean() for k in set(a) | set(b))
            return round((po - pe) / (1 - pe), 2) if pe < 1 else 1.0
        print(f"reliability on {len(M)} double-labelled episodes: "
              f"stop_reason κ={kappa(M.stop_reason, M.stop_reason_2)} ({(M.stop_reason == M.stop_reason_2).mean():.0%} agree), "
              f"fs_type κ={kappa(M.fs_type, M.fs_type_2)}, kept_trying κ={kappa(M.kept_trying, M.kept_trying_2)}, "
              f"stop attempt within 1: {((M.stop_attempt - M.stop_attempt_2).abs() <= 1).mean():.0%}")
        print(M[M.stop_reason != M.stop_reason_2][["id", "cond", "stop_reason", "stop_reason_2", "fs_type", "fs_type_2"]].to_string(index=False))
