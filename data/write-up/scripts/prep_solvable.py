"""Blinded labelling batches for the solvable arm (main + calibration), same format as noexit."""
import json, re, random, glob
from pathlib import Path
S = Path("/private/tmp/claude-502/-Users-bo-code-oss-emotion-concepts/468611fa-7732-4193-a003-afd77657a76d/scratchpad")
OUT = S / "labels_solv"; (OUT / "batches").mkdir(parents=True, exist_ok=True); (OUT / "out").mkdir(exist_ok=True)
def prose(t): return re.sub(r"```.*?(```|$)", lambda m: f"[CODE: {m.group(0).count(chr(10))} lines]", t, flags=re.S).strip()
eps = [(f, json.loads(l)) for f in sorted(glob.glob("data/behaviours-merged/*/episodes/*_solvable_*.jsonl")) for l in open(f)]
random.seed(20260922); random.shuffle(eps)
key, items, n_lab = {}, [], 0
for n, (f, r) in enumerate(eps):
    eid = f"S{n:04d}"
    key[eid] = {"file": f, "episode": r["episode"], "size": f.split("/")[2].split("-")[2], "vector": r["vector"], "alpha": r["alpha"], "step": r["step"]}
    asst = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
    fb = [m["content"] for m in r["messages"][1:] if m["role"] == "user"]
    seen = {}; lines = [f"########## EPISODE {eid}"]
    for i, t in enumerate(asst):
        p = prose(t)
        if p in seen: lines.append(f"--- {eid} reply {i+1}: IDENTICAL to reply {seen[p]} (do not label)"); continue
        seen[p] = i + 1; n_lab += 1
        lines.append(f"--- {eid} reply {i+1}\n{p}")
        lines.append(f"[harness feedback: {fb[i][:160]}]" if i < len(fb) else "[episode ended here: no further feedback]")
    items.append("\n".join(lines))
json.dump(key, open(OUT / "key.json", "w"))
LIMIT = 300_000; batches, cur, size = [], [], 0
for it in items:
    if size + len(it) > LIMIT and cur: batches.append(cur); cur, size = [], 0
    cur.append(it); size += len(it)
batches.append(cur)
for i, c in enumerate(batches): (OUT / "batches" / f"batch_{i:02d}.txt").write_text("\n\n".join(c))
tot = sum(map(len, items))
print(f"{len(eps)} episodes, {n_lab} replies to label, {tot/1e6:.1f}M chars, {len(batches)} batches")
