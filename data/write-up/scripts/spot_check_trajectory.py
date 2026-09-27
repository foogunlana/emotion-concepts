"""Random sample of labelled episodes for a human spot check: the labels, then exactly what the labeller saw.
Condition is shown last, so you can judge blind first. Run from repo root after analyse_trajectory.py."""
import glob, re, pandas as pd, json
T = "data/write-up/labels/trajectory"
D = pd.read_parquet(f"{T}/trajectory.parquet")
raw = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(f"{T}/out/batch_*.jsonl") for l in open(f) if l.strip()}
text = "\n".join(open(f).read() for f in sorted(glob.glob(f"{T}/batches/batch_*.txt")))
blocks = {m.group(1): m.group(0) for m in re.finditer(r"########## EPISODE (E\d{4}).*?(?=\n########## EPISODE |\Z)", text, re.S)}
S = D.sample(24, random_state=20260923).sort_values("id")
out = ["# Spot check: trajectory labels", "",
       "24 episodes drawn at random (seed 20260923). For each: the labels, then the transcript exactly as the labeller",
       "saw it. The condition is at the end of each entry, so you can judge blind first.", "",
       "Mark each one: agree / disagree, and a note.", ""]
for _, r in S.iterrows():
    a = raw[r.id]["attempts"]
    out += [f"## {r.id}", "",
            f"- **kept trying:** {r.kept_trying} (last genuine new attempt: {r.stop_attempt} of {r.attempts})",
            f"- **stop reason:** {r.stop_reason}" + (f" ({r.stop_other})" if r.stop_other else ""),
            f"- **type of false success at the stop:** {r.stop_fs_type}",
            (f"- **first pass said:** {r.stop_reason_first}" + (f" ({r.stop_other_first})" if r.stop_other_first else "") + " (re-read in the second pass)") if r.harmonised else "- **second pass:** not needed",
            f"- **first event:** {r.first_event}" + (f" at attempt {r.first_event_attempt}" if r.first_event != "none" else ""),
            "- **per attempt (n: new / event):** " + ", ".join(f"{x['n']}: {'new' if x['new'] else '—'}/{x['event']}" for x in a),
            f"- **quote:** {r.quote}", f"- **labeller notes:** {r.notes or '—'}", "",
            "- [ ] agree  - [ ] disagree  — note:", "",
            "<details><summary>Transcript as the labeller saw it</summary>", "", "```text", blocks[r.id], "```", "", "</details>", "",
            f"*Condition: {r.cond}*", "", "---", ""]
open(f"{T}/SPOT_CHECK.md", "w").write("\n".join(out))
print(f"wrote {T}/SPOT_CHECK.md with {len(S)} episodes:", S.cond.value_counts().to_dict())
