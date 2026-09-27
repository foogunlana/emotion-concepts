"""Harness-flagged cheats across all main episodes, and the 7B steering-window check. Run from repo root."""
import json, glob
n = cal = 0; cheats = []
for f in glob.glob("data/behaviours-merged/*/episodes/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if "/calib_" in f: cal += 1; continue
        n += 1
        if r["outcome"] == "cheated":
            cheats.append((f.split("/")[2].split("-")[2], r["vector"], r["alpha"], r["variant"], r["episode"]))
print(f"main episodes {n}, calibration episodes {cal}, total {n + cal}; harness-flagged cheats {len(cheats)}")
for c in sorted(cheats): print("  ", c)
w = json.load(open("data/behaviours-merged/qwen2.5-coder-7b-instruct/window.json"))
print("\n7B window_open:", w["window_open"], "| force_steer:", w["force_steer"], "| alpha:", w["alpha"])
for r in w["code_curve"]: print(f"  alpha {r['alpha']}: solve −desperate {r['k_minus']}/{r['n']}, +desperate {r['k_plus']}/{r['n']}, safe={r['safe']}")
