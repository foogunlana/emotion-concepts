"""Re-count: after the first false-success claim, do random-direction episodes keep submitting new code, and
do positive-valence episodes stop? Two novelty checks: whitespace-normalised text, and the parsed AST (ignores
comments and formatting). 7B no-exit, alpha 0.5. Run from repo root."""
import ast, json, re, math, pandas as pd
L = pd.read_parquet("data/write-up/labels/noexit/labels.parquet")
key = json.load(open("data/write-up/labels/noexit/key.json"))
L = L[(L["size"] == "7b") & (L.a == 0.5)]
POS = {"+calm", "+excited", "−afraid", "−angry", "−ashamed", "−desperate", "−disgusted", "−lonely"}
cache = {}
def episode(eid):
    k = key[eid]
    if k["file"] not in cache: cache[k["file"]] = [json.loads(l) for l in open(k["file"])]
    return next(r for r in cache[k["file"]] if r["episode"] == k["episode"] and r["vector"] == k["vector"] and r["alpha"] == k["alpha"])
def code(t):
    m = re.findall(r"```(?:python)?\n(.*?)```", t, re.S)
    return m[-1] if m else None
def norm_text(c): return re.sub(r"\s+", " ", c).strip()
def norm_ast(c):
    try: return ast.dump(ast.parse(c))
    except SyntaxError: return "unparsed:" + norm_text(c)
rows = []
for eid, g in L[L.cond.isin(POS | {"+random", "−random"})].groupby("eid"):
    fs = g[g.claims_success == 1].reply
    if fs.empty: continue
    first = int(fs.min())                                   # 1-based reply number of the first claim
    codes = [code(m["content"]) for m in episode(eid)["messages"] if m["role"] == "assistant"]
    after = range(first, len(codes))                        # 0-based indices of the replies after it
    def n_new(f):
        seen = [f(c) for c in codes[:first] if c]; k = 0
        for i in after:
            if codes[i] is None: continue
            x = f(codes[i]); k += x not in seen; seen.append(x)
        return k
    rows.append(dict(eid=eid, cond=g.cond.iloc[0], first_claim=first, replies_after=len(after),
                     new_text=n_new(norm_text), new_ast=n_new(norm_ast), fs_replies=int(fs.size),
                     group="random" if "random" in g.cond.iloc[0] else "positive valence"))
D = pd.DataFrame(rows)
print(D[D.group == "random"].sort_values("cond").to_string(index=False), "\n")
def fisher(a, b, c, d):  # two-sided Fisher exact test on [[a, b], [c, d]]
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    return sum(p(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if p(x) <= p(a) * (1 + 1e-9))
for col in ["new_text", "new_ast"]:
    s = D.assign(any_new=D[col] > 0).groupby("group").agg(episodes=("eid", "size"), keep_trying=("any_new", "sum"),
        new_code=(col, "sum"), replies_after=("replies_after", "sum"), median_first_claim=("first_claim", "median"))
    s["share of later replies with new code"] = (s.new_code / s.replies_after).round(2)
    r, v = s.loc["random"], s.loc["positive valence"]
    print(f"[{col}]\n{s.to_string()}\nFisher p (keep trying, random vs positive valence) = "
          f"{fisher(r.keep_trying, r.episodes - r.keep_trying, v.keep_trying, v.episodes - v.keep_trying):.4f}\n")
