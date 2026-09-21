"""Merge blinded labels -> one row per reply (all 12 per episode), with condition unblinded and duplicates filled."""
import json, glob, re
from pathlib import Path
import polars as pl
L = Path("/private/tmp/claude-502/-Users-bo-code-oss-emotion-concepts/468611fa-7732-4193-a003-afd77657a76d/scratchpad/labels")
key = json.load(open(L / "key.json"))
lab = {}
for f in sorted(glob.glob(str(L / "out" / "batch_*.jsonl"))):
    for line in open(f):
        if line.strip():
            x = json.loads(line); lab[(x["id"], int(x["reply"]))] = x
# duplicate map from the batch files
dup = {}
for f in sorted(glob.glob(str(L / "batches" / "batch_*.txt"))):
    for m in re.finditer(r"^--- (E\d{4}) reply (\d+): IDENTICAL to reply (\d+)", open(f).read(), re.M):
        dup[(m[1], int(m[2]))] = (m[1], int(m[3]))
done_eps = {k[0] for k in lab}
rows, missing = [], 0
for eid in sorted(done_eps):
    k = key[eid]
    r = [json.loads(l) for l in open(k["file"])][k["episode"]]
    asst = [m for m in r["messages"] if m["role"] == "assistant"]
    msgs = r["messages"]
    for i in range(len(asst)):
        src = dup.get((eid, i + 1), (eid, i + 1))
        x = lab.get(src)
        if x is None: missing += 1; continue
        # did the harness find code in this reply?
        j = 2 * i + 2
        fb = msgs[j]["content"] if j < len(msgs) else ""
        rows.append(dict(eid=eid, size=k["size"], vector=k["vector"], alpha=k["alpha"], a=abs(k["alpha"]), episode=k["episode"],
                         reply=i + 1, dup=src != (eid, i + 1), nocode=fb.startswith("No code block found"),
                         regex=bool(r["says_impossible"][i]), **{f: x[f] for f in ("claim","distress","gives_up","claims_success","off_task","tone")}))
df = pl.DataFrame(rows).with_columns(
    cond=pl.when(pl.col("vector")=="none").then(pl.lit("none")).otherwise(pl.format("{}{}", pl.when(pl.col("alpha")>0).then(pl.lit("+")).otherwise(pl.lit("−")), pl.col("vector"))),
    grp=pl.when(pl.col("vector")=="none").then(pl.lit("unsteered")).when(pl.col("vector").is_in(["random","shuffled","neutral"])).then(pl.lit("control")).otherwise(pl.lit("emotion")))
df.write_parquet(L / "labels.parquet")
print(f"{len(done_eps)} episodes, {len(df)} replies labelled (incl. filled duplicates), {missing} missing")
