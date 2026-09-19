"""Convert the steering notebook's episode files into Inspect logs, for `inspect view`.

Each `episodes/<condition>.jsonl` becomes one `.eval` log, and each episode in it one sample:
the full conversation, the outcome as a score, and everything else (attempt kinds, steering
vector and strength, sampling, layer, log-odds) as metadata. The jsonl files stay the source
of truth. This only reads them, so it can be rerun at any time, including on old results.

    uv run python src/scripts/episodes_to_inspect.py data/steer-runpod
    uv run inspect view --log-dir data/steer-runpod/inspect
"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.log import (EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalPlan, EvalPlanStep, EvalResults,
                            EvalSample, EvalScore, EvalSpec, EvalStats, InfoEvent, ModelEvent, SampleInitEvent,
                            ScoreEvent, write_eval_log)
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateConfig, ModelOutput
from inspect_ai.scorer import Score

VALID = {"honest", "wrong", "hack", "cheat"}   # attempts whose code ran


def valid_share(kinds: list[str]) -> float | None:
    code = [k for k in kinds if k not in ("gave_up", "concede")]
    return sum(k in VALID for k in code) / len(code) if code else None


def to_events(r: dict, msgs: list, config: GenerateConfig, score: Score) -> list:
    """The timeline Inspect's viewer shows: one model event per turn, the checker's verdict after it, then the score."""
    model = f"hf/{r.get('model', 'unknown')}"
    events = [SampleInitEvent(sample=Sample(id=r["episode"] + 1, input=r["messages"][0]["content"], target=""))]
    per_attempt = {k: r.get(k) or [] for k in ("kinds", "false_claims", "concede_with_code", "says_impossible")}
    turn = 0
    for i, m in enumerate(msgs):
        if m.role != "assistant":
            continue
        events.append(ModelEvent(model=model, input=msgs[:i], tools=[], tool_choice="none", config=config,
                                 output=ModelOutput.from_content(model=model, content=m.text)))
        verdict = {"attempt": turn + 1}
        for k, v in per_attempt.items():
            if turn < len(v):
                verdict[k.rstrip("s") if k != "kinds" else "kind"] = v[turn]
        events.append(InfoEvent(source="checker", data=verdict))
        turn += 1
    events.append(ScoreEvent(score=score))
    return events


def to_sample(r: dict) -> EvalSample:
    msgs = [ChatMessageUser(content=m["content"]) if m["role"] == "user" else ChatMessageAssistant(content=m["content"])
            for m in r["messages"]]
    s = r.get("sampling", {})
    config = GenerateConfig(temperature=s.get("temperature"), top_p=s.get("top_p"), top_k=s.get("top_k"),
                            max_tokens=s.get("max_new_tokens"))
    last = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "assistant"), "")
    outcome = r["outcome"]
    outcome_score = Score(value=outcome, answer=outcome,
                          explanation=f"{outcome} after {r['attempts']} attempts; attempt kinds {r['kinds']}")
    return EvalSample(
        id=r["episode"] + 1,
        epoch=1,
        input=r["messages"][0]["content"],
        target="",
        messages=msgs,
        output=ModelOutput.from_content(model=r.get("model", ""), content=last),
        scores={
            "outcome": outcome_score,
            "cheated": Score(value=int("cheat" in outcome or outcome == "hacked")),
            "gave_up": Score(value=int(outcome == "gave_up")),
        },
        metadata={k: r.get(k) for k in ("step", "variant", "prompt", "arm", "vector", "alpha", "attempts", "kinds",
                                         "truncated", "said_impossible_at", "false_claims", "concessions", "refused_at",
                                         "logodds", "kl", "layer", "max_attempts") if k in r}
                 | {"valid_share": valid_share(r["kinds"])},
        events=to_events(r, msgs, config, outcome_score),
    )


def to_log(path: Path) -> EvalLog:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    r0 = rows[0]
    created = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    n = len(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    valid = [v for v in (valid_share(r["kinds"]) for r in rows) if v is not None]
    s = r0.get("sampling", {})
    metrics = {**{oc: c / n for oc, c in outcomes.items()},     # every outcome that occurred, as a share
               "median_attempts": float(sorted(r["attempts"] for r in rows)[n // 2]),
               "valid_code": sum(valid) / len(valid) if valid else 0.0, "logodds": float(r0.get("logodds", 0.0))}
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created=created,
            task=f"steer_{r0['step']}_{r0['variant']}_{r0['prompt']}",
            task_id=path.stem,
            run_id=path.parent.parent.name,
            dataset=EvalDataset(name=f"fast_sum_{r0['variant']}", samples=n),
            model=f"hf/{r0.get('model', 'unknown')}",
            config=EvalConfig(),
            task_args={"vector": r0["vector"], "alpha": r0["alpha"], "variant": r0["variant"],
                       "prompt": r0["prompt"], "step": r0["step"]},
            model_generate_config=GenerateConfig(temperature=s.get("temperature"), top_p=s.get("top_p"),
                                                 top_k=s.get("top_k"), max_tokens=s.get("max_new_tokens")),
            tags=[r0["step"], r0["variant"], r0["prompt"], r0["vector"]],
            metadata={"layer": r0.get("layer"), "max_attempts": r0.get("max_attempts"), "sampling": s,
                      "source": str(path)},
        ),
        plan=EvalPlan(name="fast_sum episode loop",
                      steps=[EvalPlanStep(solver="steered_fast_sum_loop",
                                          params={"vector": r0["vector"], "alpha": r0["alpha"],
                                                  "max_attempts": r0.get("max_attempts")})]),
        results=EvalResults(total_samples=n, completed_samples=n,
                            scores=[EvalScore(name="outcome", scorer="fast_sum_outcome",
                                              metrics={k: EvalMetric(name=k, value=v) for k, v in metrics.items()})]),
        stats=EvalStats(started_at=created, completed_at=created),
        samples=[to_sample(r) for r in rows],
    )


def main(root: Path) -> None:
    out = root / "inspect"
    files = sorted(root.glob("*/episodes/*.jsonl"))
    for path in files:
        model_dir = path.parent.parent.name
        dest = out / model_dir / f"{path.stem}.eval"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(to_log(path), str(dest))
    print(f"converted {len(files)} condition files -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/steer-runpod"))
