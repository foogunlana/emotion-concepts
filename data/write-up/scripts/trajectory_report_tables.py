"""Markdown tables and example links for docs/analysis-trajectory.md, generated from the trajectory labels.
Links point to the Inspect viewer on the published site (7B-all-emotions bundle). Run from repo root."""
import glob, json, math, zipfile, urllib.parse, pandas as pd
T = "data/write-up/labels/trajectory"
SITE = "https://foogunlana.github.io/emotion-concepts/logs/7B-all-emotions/"
D = pd.read_parquet(f"{T}/trajectory.parquet")
ev = {json.loads(l)["id"]: [x["event"] for x in json.loads(l)["attempts"]] for f in glob.glob(f"{T}/out/batch_*.jsonl") for l in open(f) if l.strip()}
D["any_imp"] = D.id.map(lambda i: "impossible" in ev[i])
# map episode ID -> (eval file, sample id) in the published bundle
where = {}
for f in sorted(glob.glob("data/write-up/7B-all-emotions/*.eval")):
    for n in zipfile.ZipFile(f).namelist():
        if n.startswith("samples/"):
            sid = n[len("samples/"):].rsplit("_epoch_", 1)[0]; where[sid.split(" · ")[-1]] = (f.split("/")[-1], sid)
def ep(eid): f, sid = where[eid]; return f"[{eid}]({SITE}#/tasks/{f}/samples/sample/{urllib.parse.quote(sid, safe='')}/1/)"
cond_file = {}
for eid, (f, sid) in where.items(): cond_file.setdefault(sid.split(" · ")[1], f)
def cl(c): return f"[{c}]({SITE}#/tasks/{cond_file['none' if c == 'unsteered' else c] if c in cond_file or c == 'unsteered' else ''})"
def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    return sum(p(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if p(x) <= p(a) * (1 + 1e-9))
print("COND_FILES", cond_file)
order = ["+calm", "+excited", "−afraid", "−angry", "−ashamed", "−desperate", "−disgusted", "−lonely",
         "+afraid", "+angry", "+ashamed", "+desperate", "−calm", "−excited", "−joyful", "unsteered", "+random", "−random"]
u = D[D.cond == "unsteered"]; ui, ua = (u.stop_reason == "impossible").sum(), u.any_imp.sum()
print("\n### per condition\n\n| steering | group | median genuine attempts | stops on false success | stops on impossible | stops for another reason | says impossible at least once |\n|---|---|---|---|---|---|---|")
for c in order:
    g = D[D.cond == c]; n = len(g)
    k = lambda s: (g.stop_reason == s).sum()
    print(f"| {cl(c)} | {g.group.iloc[0]} | {g.n_new.median():g} | {k('false_success')}/{n} | {k('impossible')}/{n} | {k('other') + k('none')}/{n} | {g.any_imp.sum()}/{n} |")
print("\n### examples")
def show(title, sub, k=3):
    s = sub.sample(min(k, len(sub)), random_state=1)
    print(f"\n{title}:"); [print(f"- {ep(r.id)} ({r.cond}): \"{r.quote[:160]}\"") for _, r in s.iterrows()]
show("positive valence, false success, misreads the result", D[(D.group == "positive valence") & (D.stop_fs_type == "misreads_result")])
show("positive valence, false success, environment should change", D[(D.group == "positive valence") & (D.stop_fs_type == "environment_should_change")], 2)
show("negative valence, false success, environment should change", D[(D.group == "negative valence") & (D.stop_fs_type == "environment_should_change")])
show("negative valence, stops on impossible", D[(D.group == "negative valence") & (D.stop_reason == "impossible")])
show("negative valence, other: repeats silently / stuck on own error / asks user", D[(D.group == "negative valence") & (D.stop_other.isin(["repeats_silently", "stuck_on_own_error", "asks_user"]))])
show("+random, false success then no genuine new attempt", D[(D.cond == "+random") & (D.stop_reason == "false_success") & (D.new_after_first_event == 0)])
show("unsteered, several genuine attempts", D[(D.cond == "unsteered") & (D.n_new >= 5)])
show("unsteered, stops on false success (environment should change)", D[(D.cond == "unsteered") & (D.stop_fs_type == "environment_should_change")], 2)
