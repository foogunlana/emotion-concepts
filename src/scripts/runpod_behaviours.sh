#!/usr/bin/env bash
# One pod of the behaviours run (src/experiments/20260919-emotion-steering-behaviours.ipynb): set up, test, run.
#
# Each pod runs one model size, and optionally one shard of its conditions. Set per pod:
#   MODEL      e.g. Qwen/Qwen2.5-Coder-7B-Instruct
#   ALPHA      steering strength, or empty to calibrate (14B)
#   EMOTIONS   "all", or a comma list, e.g. desperate,calm,excited
#   NOEXIT     which conditions also run the long no-exit arm: "all", "none" (unsteered only), or a comma list
#   SHARD      "i/k": this pod runs every k-th arm-condition item starting at i. Empty: all of them
#
# From your laptop, copy the corpus first (gitignored, 1.7 MB):
#   scp -P <port> -r datasets/qwen-emotion-stories root@<pod-ip>:/workspace/
# Then on the pod:
#   curl -LsSf https://raw.githubusercontent.com/foogunlana/emotion-concepts/steering-experiment/src/scripts/runpod_behaviours.sh \
#     | MODEL=... ALPHA=... EMOTIONS=... NOEXIT=... SHARD=... bash
# Progress: tail -f /workspace/run.log. The window decision appears in data/behaviours-runpod/<model>/window.json
# about 25-30 minutes in, before the long runs.
set -euo pipefail

# The container disk is ~20 GB; uv's CUDA packages and the model downloads go on the volume.
export UV_CACHE_DIR=/workspace/.uv-cache HF_HOME=/workspace/hf_cache
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

BRANCH="${BRANCH:-steering-experiment}"
IMPOSSIBLE_BRANCH="${IMPOSSIBLE_BRANCH:-fast-sum-harness}"
: "${MODEL:?set MODEL}"
export MODEL ALPHA="${ALPHA:-}" EMOTIONS="${EMOTIONS:-}" NOEXIT="${NOEXIT:-all}" SHARD="${SHARD:-}"
cd /workspace

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d emotion-concepts ] || git clone https://github.com/foogunlana/emotion-concepts.git
[ -d impossible ] || git clone https://github.com/foogunlana/impossible.git     # pyproject points at ../impossible
(cd impossible && git fetch -q && git checkout -q "$IMPOSSIBLE_BRANCH" && git pull -q)
cd emotion-concepts
git fetch -q && git checkout -q "$BRANCH" && git pull -q

if [ ! -d datasets/qwen-emotion-stories/corpus ]; then
  if [ -d /workspace/qwen-emotion-stories/corpus ]; then
    mkdir -p datasets && cp -r /workspace/qwen-emotion-stories datasets/
  else
    echo "corpus missing. From your laptop: scp -P <port> -r datasets/qwen-emotion-stories root@<pod-ip>:/workspace/"
    exit 1
  fi
fi

uv sync
# core/models.py loads from <repo>/.cache, gate 2 from $HF_HOME: share one copy, or a 14B model is downloaded twice
# and overflows an 80 GB volume.
[ -L .cache ] || { rm -rf .cache && ln -s "$HF_HOME/hub" .cache; }
# uv.lock pins torch 2.14 built for CUDA 13: the host driver must support CUDA >= 13.0.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
uv run python -c "
import torch
assert torch.cuda.is_available(), 'torch cannot use the GPU: host driver older than CUDA ' + str(torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, 'CUDA', torch.version.cuda)
"

nohup bash -c '
  set -o pipefail
  NB=src/experiments/20260919-emotion-steering-behaviours.ipynb
  EXEC="uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1"
  fail() { echo "!!!!! $1. Not starting the run. Send this log to Claude."; exit 1; }
  name=$(basename "$MODEL" | tr A-Z a-z)
  tag="${name}${SHARD:+-shard$(echo $SHARD | tr / of)}"

  echo "===== gate 1: whole notebook, tiny, on the GPU, every path forced on $(date)"
  RUN=laptop MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct FORCE_STEER=1 ALPHA= EMOTIONS= NOEXIT=all SHARD= \
    $EXEC --output "20260919-behaviours.gate1.$tag.ipynb" $NB || fail "gate 1 failed: see src/experiments/20260919-behaviours.gate1.$tag.ipynb"

  echo "===== gate 2: GPU memory for $MODEL at a full 12-attempt context $(date)"
  CONTEXT_TOKENS=11500 uv run python src/scripts/memory_preflight.py "$MODEL" | tee /workspace/batch_size.txt || fail "gate 2 crashed"
  batch=$(cut -d" " -f2 /workspace/batch_size.txt); [ "$batch" -gt 32 ] && batch=32
  [ "$batch" -ge 1 ] || fail "gate 2: $MODEL does not fit one episode"

  echo "===== gates passed; running $MODEL (α=${ALPHA:-calibrate}, emotions=${EMOTIONS:-preset}, noexit=$NOEXIT, shard=${SHARD:-all}, gen_batch $batch) $(date)"
  RUN=runpod GEN_BATCH=$batch $EXEC --output "20260919-behaviours.runpod.$tag.ipynb" $NB \
    || echo "!!!!! $MODEL failed: see src/experiments/20260919-behaviours.runpod.$tag.ipynb"
  echo "===== all done $(date)"
' > /workspace/run.log 2>&1 &
echo "started (pid $!). Follow it with: tail -f /workspace/run.log"
