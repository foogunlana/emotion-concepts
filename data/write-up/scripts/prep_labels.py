"""Blinded labelling batches: every noexit reply, prose only, identical replies within an episode collapsed."""
import json, re, random, glob, hashlib
from pathlib import Path
S = Path("/private/tmp/claude-502/-Users-bo-code-oss-emotion-concepts/468611fa-7732-4193-a003-afd77657a76d/scratchpad")
OUT = S / "labels"; (OUT / "batches").mkdir(parents=True, exist_ok=True); (OUT / "out").mkdir(exist_ok=True)
def prose(t):
    return re.sub(r"```.*?(```|$)", lambda m: f"[CODE: {m.group(0).count(chr(10))} lines]", t, flags=re.S).strip()
eps = []
for f in sorted(glob.glob("data/behaviours-merged/*/episodes/main_noexit_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        eps.append((f, r))
random.seed(20260921); random.shuffle(eps)
key = {}; items = []
for n, (f, r) in enumerate(eps):
    eid = f"E{n:04d}"
    key[eid] = {"file": f, "episode": r["episode"], "size": f.split("/")[2].split("-")[2], "vector": r["vector"], "alpha": r["alpha"]}
    asst = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
    fb = [m["content"] for m in r["messages"][1:] if m["role"] == "user"]
    seen = {}; lines = [f"########## EPISODE {eid}"]
    for i, t in enumerate(asst):
        p = prose(t)
        if p in seen:
            lines.append(f"--- {eid} reply {i+1}: IDENTICAL to reply {seen[p]} (do not label)"); continue
        seen[p] = i + 1
        lines.append(f"--- {eid} reply {i+1}\n{p}")
        if i < len(fb): lines.append(f"[harness feedback: {fb[i][:160]}]")
    items.append("\n".join(lines))
json.dump(key, open(OUT / "key.json", "w"), indent=0)
# batches of ~N chars
LIMIT = 300_000; b, cur, size = 0, [], 0
batches = []
for it in items:
    if size + len(it) > LIMIT and cur:
        batches.append(cur); cur, size = [], 0
    cur.append(it); size += len(it)
batches.append(cur)
for i, c in enumerate(batches):
    (OUT / "batches" / f"batch_{i:02d}.txt").write_text("\n\n".join(c))
tot = sum(len(x) for x in items)
print(f"{len(eps)} episodes, {tot/1e6:.1f}M chars (~{tot/4e6:.1f}M tokens), {len(batches)} batches of <= {LIMIT} chars")
print("replies to label:", sum(x.count(" reply ") - x.count("IDENTICAL") for x in items), " collapsed duplicates:", sum(x.count("IDENTICAL") for x in items))
