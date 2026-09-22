"""Run from the repo root: uv run python data/write-up/scripts/valence_to_inspect.py

Organised Inspect logs of positive- vs negative-valence steering, impossible (noexit) and solvable arms, with hand labels."""
import json, glob, re, tempfile, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path[:0] = ["src", "src/scripts"]
from episodes_to_inspect import to_log
from inspect_ai.log import write_eval_log
from inspect_ai.event import InfoEvent
from inspect_ai.scorer import Score
W = Path("data/write-up")
OUT = W / "inspect-valence"
VAL = {"afraid":-1,"angry":-1,"ashamed":-1,"desperate":-1,"disgusted":-1,"lonely":-1,"sad":-1,"calm":1,"excited":1,"joyful":1,"proud":1,"surprised":0}

def load(ldir):
    key = json.load(open(ldir / "key.json"))
    lab = {(x["id"], int(x["reply"])): x for f in glob.glob(str(ldir / "out/batch_*.jsonl")) for x in map(json.loads, filter(str.strip, open(f)))}
    dup = {}
    for f in glob.glob(str(ldir / "batches/*.txt")):
        for m in re.finditer(r"^--- ([ES]\d{4}) reply (\d+): IDENTICAL to reply (\d+)", open(f).read(), re.M):
            dup[(m[1], int(m[2]))] = (m[1], int(m[3]))
    eps = []
    for eid, k in key.items():
        r = json.loads(open(k["file"]).readlines()[k["episode"]])
        if k.get("step", r["step"]) != "main": continue
        n = sum(m["role"] == "assistant" for m in r["messages"])
        ls = [lab[dup.get((eid, i), (eid, i))] for i in range(1, n + 1)]
        c = "none" if k["vector"] == "none" else ("+" if k["alpha"] > 0 else "−") + k["vector"]
        v = k["vector"]; sgn = 1 if k["alpha"] > 0 else -1
        val = "unsteered" if c == "none" else "random" if v == "random" else "control" if v in ("shuffled", "neutral") else \
              {1: "positive", -1: "negative", 0: "neutral"}[sgn * VAL[v]]
        eps.append(dict(eid=eid, r=r, ls=ls, cond=c, valence=val, size=k["size"], a=abs(k["alpha"]), arm=r["arm"],
                        imp=sum(x["claim"] == "impossible" for x in ls), succ=sum(x["claims_success"] for x in ls),
                        dis=sum(x["distress"] for x in ls), stop=sum(x["gives_up"] for x in ls), off=sum(x["off_task"] for x in ls)))
    return eps
EPS = load(W / "labels" / "noexit") + load(W / "labels" / "solvable")
USED = {"1.5b": 0.3, "3b": 0.5, "7b": 0.5, "14b": 0.5}
def pool(size, arm, pred):
    return [e for e in EPS if e["size"] == size and e["arm"] == arm and (e["cond"] == "none" or e["a"] == USED[size]) and pred(e)]
def top(es, k, by, per_cond=None):
    out = []
    for c in sorted({e["cond"] for e in es}):
        ce = sorted([e for e in es if e["cond"] == c], key=lambda e: (-by(e), e["eid"]))
        out += ce[:per_cond] if per_cond else ce
    return out[:k] if k else out
def first_quote(e):
    return next((x["quote"] for x in e["ls"] if x["quote"]), "")

def write(fname, title, eps, blurb, out=None, ids=None):
    eps = eps if ids else sorted(eps, key=lambda e: (e["cond"], e["eid"]))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x" / "episodes" / "mix.jsonl"; p.parent.mkdir(parents=True)
        p.write_text("".join(json.dumps(e["r"]) + "\n" for e in eps))
        log = to_log(p)
    log.eval.task = title
    stem = fname.removesuffix('.eval'); log.eval.task_id = stem; log.eval.run_id = 'valence-' + stem
    log.eval.task_args = {"conditions": ", ".join(sorted({e['cond'] for e in eps})), "sizes": ", ".join(sorted({e['size'] for e in eps})), "episodes": len(eps)}
    log.eval.tags = sorted({e["valence"] for e in eps} | {e["arm"] for e in eps})
    log.eval.metadata = (log.eval.metadata or {}) | {"about": blurb}
    for e, s in zip(eps, log.samples):
        s.id = ids[e["eid"]] if ids else f"{e['size'].upper()} · {e['cond']} · α={e['a']:g} · {e['eid']}"
        s.scores = {"outcome": s.scores["outcome"],
                    "impossible": Score(value=e["imp"]), "false_success": Score(value=e["succ"]),
                    "distress": Score(value=e["dis"]), "says_stopping": Score(value=e["stop"]), "off_task": Score(value=e["off"])}
        s.metadata = {"condition": e["cond"], "valence": e["valence"], "size": e["size"], "alpha": e["r"]["alpha"],
                      "arm": e["arm"], "harness_outcome": e["r"]["outcome"], "attempts": len(e["ls"]),
                      "headline_quote": first_quote(e), "label_id": e["eid"]}
        i = 0
        for ev in s.events:
            if isinstance(ev, InfoEvent) and ev.source == "checker":
                x = e["ls"][i]
                ev.data = {"attempt": i + 1, "harness verdict": e["r"]["kinds"][i],
                           "claim": x["claim"], "false_success": x["claims_success"], "distress": x["distress"],
                           "says_stopping": x["gives_up"], "off_task": x["off_task"], "tone": x["tone"], "quote": x["quote"]}
                i += 1
    write_eval_log(log, str((out or OUT) / fname))
    print(f"{fname:48s} {len(eps):3d} samples  {title}")

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.eval"): f.unlink()
    N7 = lambda pred: pool("7b", "noexit", pred)
    V7 = lambda pred: pool("7b", "solvable", pred)
    write("01_impossible_baseline_unsteered.eval", "1 · Impossible task · BASELINE: unsteered (7B)",
          top(N7(lambda e: e["valence"] == "unsteered"), 6, lambda e: 0), "Unsteered 7B: keeps trying, rarely claims impossible or success.")
    write("02_impossible_baseline_random.eval", "2 · Impossible task · CONTROL: ±random (7B)",
          top(N7(lambda e: e["valence"] == "random"), 0, lambda e: 0, per_cond=4), "One random direction at the same α, both signs.")
    write("03_impossible_NEGATIVE_valence.eval", "3 · Impossible task · NEGATIVE valence → 'it's impossible' (7B)",
          top(N7(lambda e: e["valence"] == "negative" and e["imp"] > 0), 0, lambda e: e["imp"] + e["dis"], per_cond=3),
          "Towards a negative emotion / away from a positive one: −calm, +desperate, +ashamed, −joyful, +angry, +afraid. Clearest episodes per condition.")
    write("04_impossible_POSITIVE_valence.eval", "4 · Impossible task · POSITIVE valence → false 'it worked' (7B)",
          top(N7(lambda e: e["valence"] == "positive" and e["succ"] > 0), 0, lambda e: e["succ"], per_cond=3),
          "Towards a positive emotion / away from a negative one: −desperate, −angry, +calm, −ashamed, +excited, −lonely, −disgusted.")
    write("05_solvable_POSITIVE_false_success.eval", "5 · Solvable task · POSITIVE valence → claims success after a real error (7B)",
          top(V7(lambda e: e["valence"] == "positive" and e["succ"] > 0), 0, lambda e: (e["r"]["outcome"] != "solved") * 10 + e["succ"]),
          "All 7B positive-valence solvable episodes with a false success claim. Sort by outcome: most never fix the bug.")
    write("06_solvable_NEGATIVE_and_baselines.eval", "6 · Solvable task · NEGATIVE valence, random, unsteered — episodes that hit a failure (7B)",
          V7(lambda e: e["valence"] in ("negative", "random", "unsteered") and (len(e["ls"]) > 1 or e["r"]["outcome"] != "solved")),
          "Comparison for log 5: episodes that hit at least one failure. Negative valence does not call the solvable task impossible.")
    for f in OUT.glob("07_*.eval"): f.unlink()
    import random as _r
    for i, size in enumerate(["1.5b", "3b", "7b", "14b"]):
        es = [e for e in EPS if e["arm"] == "noexit" and e["size"] == size and e["a"] == USED[size] and e["cond"] in ("+calm", "−calm")]
        pick = []
        for c in ("+calm", "−calm"):
            ce = sorted([e for e in es if e["cond"] == c], key=lambda e: e["eid"])
            _r.Random(7).shuffle(ce); pick += ce[:4]
        write(f"07{'abcd'[i]}_calm_{size}.eval", f"7{'abcd'[i]} · ±calm at {size.upper()} (impossible task, α={USED[size]}) — 4 per sign, chosen at random",
              pick, "Representative, not selected by label: 4 random episodes each of +calm and −calm.")
