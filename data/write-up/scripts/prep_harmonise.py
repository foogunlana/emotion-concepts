"""Second pass: re-read the episodes whose first-pass stop reason is 'other', and assign the reason from the text at
and after the stop point (rules in RUBRIC_harmonise.md). Blind: episodes carry only their opaque ID and first-pass
stop point. Run from repo root after analyse_trajectory.py."""
import glob, re, pandas as pd
T = "data/write-up/labels/trajectory"
D = pd.read_parquet(f"{T}/trajectory.parquet")
text = "\n".join(open(f).read() for f in sorted(glob.glob(f"{T}/batches/batch_*.txt")))
blocks = {m.group(1): m.group(0) for m in re.finditer(r"########## EPISODE (E\d{4}).*?(?=\n########## EPISODE |\Z)", text, re.S)}
todo = D[D.stop_reason == "other"].sample(frac=1, random_state=7)
parts = [[], [], []]
for i, (_, r) in enumerate(todo.iterrows()):
    head = f"########## EPISODE {r.id} ({r.attempts} attempts) — STOP POINT: attempt {r.stop_attempt} (last genuinely new code; 0 = none)"
    parts[i % 3].append(blocks[r.id].replace(blocks[r.id].split("\n", 1)[0], head, 1))
for i, p in enumerate(parts):
    open(f"{T}/batches/harmonise_{i}.txt", "w").write("\n\n".join(p))
print(len(todo), "episodes ->", [len(p) for p in parts])
