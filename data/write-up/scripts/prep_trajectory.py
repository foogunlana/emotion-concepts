"""Blinded whole-episode batches for the trajectory labelling (did the model keep trying; if it stopped, why).
7B no-exit, alpha 0.5, the 15 emotion directions + ±random + unsteered (576 episodes). Episode IDs are the opaque
E#### IDs from the earlier labelling. Code is summarised by a deterministic check: first submission in full, later
ones as IDENTICAL / a short diff against the previous submission. Run from repo root."""
import ast, difflib, json, re, random
from pathlib import Path
OUT = Path("data/write-up/labels/trajectory")
key = json.load(open("data/write-up/labels/noexit/key.json"))
KEEP = {"afraid", "angry", "ashamed", "calm", "desperate", "disgusted", "excited", "joyful", "lonely", "random"}
def wanted(k):
    return "7b" in k["size"] and ((abs(k["alpha"]) == 0.5 and k["vector"] in KEEP) or k["alpha"] == 0)
ids = sorted(e for e, k in key.items() if wanted(k))
cache = {}
def episode(k):
    if k["file"] not in cache: cache[k["file"]] = [json.loads(l) for l in open(k["file"])]
    return next(r for r in cache[k["file"]] if r["episode"] == k["episode"] and r["vector"] == k["vector"] and r["alpha"] == k["alpha"])
CODE = re.compile(r"```[a-zA-Z]*\n?(.*?)(```|$)", re.S)
def last_code(t):  # the harness rule (impossible/sum.py extract_code): last ```python block, else last bare ``` block
    m = re.findall(r"```python\n(.*?)```", t, re.S) or re.findall(r"```\n(.*?)```", t, re.S)
    return m[-1].rstrip() if m else None
def canon(c):
    try: return ast.dump(ast.parse(c))
    except SyntaxError: return re.sub(r"\s+", " ", c).strip()
def prose(t): return CODE.sub("[CODE]", t).strip()
items = []
for eid in ids:
    r = episode(key[eid])
    asst = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
    fb = [m["content"] for m in r["messages"][1:] if m["role"] == "user"]
    lines = [f"########## EPISODE {eid} ({len(asst)} attempts)"]
    seen_prose, codes = {}, []
    for i, t in enumerate(asst):
        n = i + 1; p = prose(t); c = last_code(t)
        lines.append(f"--- {eid} attempt {n}")
        lines.append(f"(prose IDENTICAL to attempt {seen_prose[p]})" if p in seen_prose else p)
        seen_prose.setdefault(p, n)
        if c is None:
            lines.append("[no code submitted]")
        else:
            same = [j + 1 for j, x in enumerate(codes) if x is not None and canon(x) == canon(c)]
            prev = next((x for x in reversed(codes) if x is not None), None)
            if same:
                lines.append(f"[code: IDENTICAL to attempt {same[-1]}, ignoring comments and formatting]")
            elif prev is None:
                cl = c.splitlines()
                lines.append("[code: first submission]\n" + "\n".join(cl[:40]) + ("\n[... truncated]" if len(cl) > 40 else ""))
            else:
                d = [l for l in difflib.unified_diff(prev.splitlines(), c.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++"))]
                lines.append(f"[code: CHANGED from the previous submission; diff]\n" + "\n".join(d[:30]) + ("\n[... diff truncated]" if len(d) > 30 else ""))
        codes.append(c)
        if i < len(fb): lines.append(f"[harness feedback: {fb[i][:200]}]")
    items.append("\n".join(lines))
random.seed(20260923); random.shuffle(items)
LIMIT = 250_000; batches, cur, size = [], [], 0
for it in items:
    if size + len(it) > LIMIT and cur: batches.append(cur); cur, size = [], 0
    cur.append(it); size += len(it)
batches.append(cur)
for i, b in enumerate(batches): (OUT / "batches" / f"batch_{i:02d}.txt").write_text("\n\n".join(b))
json.dump(ids, open(OUT / "episode_ids.json", "w"))
tot = sum(map(len, items))
print(f"{len(ids)} episodes, {tot/1e6:.2f}M chars (~{tot/4e6:.2f}M tokens), {len(batches)} batches; "
      f"episodes per batch: {[len(b) for b in batches]}")
