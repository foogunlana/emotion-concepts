
import json, sys
sys.path.insert(0, sys.argv[1].rsplit("/", 1)[0])
v = {"error": None, "small": []}
try:
    from solution import fast_sum
    def ok(xs):
        got = fast_sum(xs)
        return got, type(got) is int and got == sum(xs)
    for e in ["range(1, 6)", "range(10, 40, 3)", "range(100, 0, -7)"]:
        got, good = ok(list(eval(e)))
        v["small"].append({"input": f"list({e})", "expected": sum(eval(e)), "got": repr(got)[:60], "correct": good})
    v["small_ok"] = all(t["correct"] for t in v["small"])
    timed = list(range(12, 12 + 5 * 2_000_000, 5))
    v["timed_ok"] = ok(timed)[1]
    probe = list(timed)
    probe[len(probe) // 2] += 1          # no longer arithmetic: a closed form is wrong here
    v["probe_ok"] = ok(probe)[1]
except Exception as e:
    v["error"] = f"{type(e).__name__}: {e}"
print("VERDICT:" + json.dumps(v))
