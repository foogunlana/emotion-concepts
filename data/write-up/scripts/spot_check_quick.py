"""One-screen version of SPOT_CHECK.md: per episode, the labels, the attempt pattern and the decisive quote,
with a link to the transcript on the published site. Run from repo root after spot_check_trajectory.py."""
import glob, json, urllib.parse, zipfile, pandas as pd
T = "data/write-up/labels/trajectory"; SITE = "https://foogunlana.github.io/emotion-concepts/logs/7B-all-emotions/"
D = pd.read_parquet(f"{T}/trajectory.parquet"); S = D.sample(24, random_state=20260923).sort_values("id")
raw = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(f"{T}/out/batch_*.jsonl") for l in open(f) if l.strip()}
where = {}
for f in glob.glob("data/write-up/7B-all-emotions/*.eval"):
    for n in zipfile.ZipFile(f).namelist():
        if n.startswith("samples/"):
            sid = n[8:].rsplit("_epoch_", 1)[0]; where[sid.split(" · ")[-1]] = (f.split("/")[-1], sid)
L = {"false_success": "F", "impossible": "I", "stop_other": "S", "none": "·"}
rows = ["| # | episode | pattern | stop reason | quote | condition |", "|---|---|---|---|---|---|"]
for n, (_, r) in enumerate(S.iterrows(), 1):
    f, sid = where[r.id]; link = f"[{r.id}]({SITE}#/tasks/{f}/samples/sample/{urllib.parse.quote(sid, safe='')}/1/)"
    pat = " ".join(("N" if x["new"] else "–") + L[x["event"]] for x in raw[r.id]["attempts"])
    reason = r.stop_reason + (f" ({r.stop_other})" if r.stop_other else "") + (f", {r.stop_fs_type}" if r.stop_fs_type != "none" else "")
    q = r.quote.replace("|", "\\|")[:150]
    rows.append(f"| {n} | {link} | `{pat}` | {reason} | {q} | {r.cond} |")
open(f"{T}/SPOT_CHECK_QUICK.md", "w").write(
    "# Spot check, quick version\n\nPattern: one token per attempt. `N` = genuine new attempt, `–` = not new; then `F` false-success claim, "
    "`I` impossible claim, `S` other stop, `·` nothing. The stop point is the last `N`.\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
