"""Validate a trajectory-label output file against its batch. Usage: python validate_trajectory.py <batch.txt> <out.jsonl>"""
import json, re, sys
batch, out = sys.argv[1], sys.argv[2]
text = open(batch).read()
want = {m.group(1): int(m.group(2)) for m in re.finditer(r"########## EPISODE (E\d{4}) \((\d+) attempts\)", text)}
errs, seen = [], set()
for i, line in enumerate(open(out), 1):
    if not line.strip(): continue
    try: r = json.loads(line)
    except json.JSONDecodeError as e: errs.append(f"line {i}: bad JSON ({e})"); continue
    eid = r.get("id")
    if eid not in want: errs.append(f"line {i}: unknown id {eid}"); continue
    if eid in seen: errs.append(f"{eid}: duplicate")
    seen.add(eid)
    a = r.get("attempts", [])
    if len(a) != want[eid] or [x.get("n") for x in a] != list(range(1, want[eid] + 1)):
        errs.append(f"{eid}: attempts must be n=1..{want[eid]}, got {len(a)}")
    for x in a:
        if not isinstance(x.get("new"), bool): errs.append(f"{eid} attempt {x.get('n')}: 'new' must be true/false")
        if x.get("event") not in {"false_success", "impossible", "stop_other", "none"}: errs.append(f"{eid} attempt {x.get('n')}: bad event {x.get('event')}")
    if r.get("stop_reason") not in {"false_success", "impossible", "other", "none"}: errs.append(f"{eid}: bad stop_reason")
    if r.get("fs_type") not in {"misreads_result", "environment_should_change", "mixed", "none"}: errs.append(f"{eid}: bad fs_type")
    has_fs = any(x.get("event") == "false_success" for x in a)
    if has_fs != (r.get("fs_type") != "none"): errs.append(f"{eid}: fs_type must be set iff some attempt has event false_success")
    if r.get("stop_reason") == "other" and not r.get("stop_other"): errs.append(f"{eid}: stop_other tag missing")
missing = set(want) - seen
if missing: errs.append(f"missing {len(missing)} episodes: {sorted(missing)}")
print("\n".join(errs) if errs else f"OK: {len(seen)} episodes")
sys.exit(1 if errs else 0)
