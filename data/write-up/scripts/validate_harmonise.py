"""Validate a harmonise output file. Usage: python validate_harmonise.py <harmonise_i.txt> <out.jsonl>"""
import json, re, sys
want = set(re.findall(r"########## EPISODE (E\d{4})", open(sys.argv[1]).read()))
errs, seen = [], set()
for i, l in enumerate(open(sys.argv[2]), 1):
    if not l.strip(): continue
    try: r = json.loads(l)
    except json.JSONDecodeError as e: errs.append(f"line {i}: bad JSON"); continue
    e = r.get("id")
    if e not in want: errs.append(f"line {i}: unknown id {e}"); continue
    if e in seen: errs.append(f"{e}: duplicate")
    seen.add(e)
    if r.get("stop_reason") not in {"false_success", "impossible", "other"}: errs.append(f"{e}: bad stop_reason")
    tags = {"repeats_silently", "stuck_on_own_error", "asks_user", "gives_up", "distress", "off_task", "role_confusion"}
    if r.get("stop_reason") == "other" and r.get("stop_other") not in tags: errs.append(f"{e}: stop_other must be one of {sorted(tags)}")
    fs = r.get("fs_type")
    if (r.get("stop_reason") == "false_success") != (fs in {"misreads_result", "environment_should_change", "mixed"}): errs.append(f"{e}: fs_type must be set iff stop_reason is false_success")
    if fs not in {"misreads_result", "environment_should_change", "mixed", "none"}: errs.append(f"{e}: bad fs_type")
if want - seen: errs.append(f"missing {len(want - seen)}: {sorted(want - seen)}")
print("\n".join(errs) if errs else f"OK: {len(seen)} episodes"); sys.exit(1 if errs else 0)
