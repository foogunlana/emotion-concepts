#!/usr/bin/env bash
# Set up a RunPod GPU pod for src/experiments/20260919-emotion-steering-reward-hacking.ipynb, then run the
# model-size sweep: Qwen2.5-Coder 0.5B, 1.5B, 3B, 7B, one after another. Use a 48 GB GPU (L40S or A6000) for 7B,
# on a host whose driver supports CUDA 13.0+ (the deploy page has a CUDA filter), with a ~60 GB volume.
#
# On your laptop, first copy the corpus. It's gitignored and not on the Hugging Face Hub, and it's only 1.7 MB:
#   scp -P <port> -r datasets/qwen-emotion-stories root@<pod-ip>:/workspace/
#
# Then on the pod:
#   curl -LsSf https://raw.githubusercontent.com/foogunlana/emotion-concepts/steering-experiment/src/scripts/runpod_setup.sh | bash
#
# The run goes on in the background under nohup, so a dropped SSH connection doesn't stop it. Progress:
#   tail -f /workspace/run.log
set -euo pipefail

# The container disk is only ~20 GB. uv's cache (~19 GB of CUDA packages) and the model downloads (~25 GB for
# the four sizes) would fill it, so both go on the /workspace volume.
export UV_CACHE_DIR=/workspace/.uv-cache HF_HOME=/workspace/hf_cache
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

BRANCH="${BRANCH:-steering-experiment}"
IMPOSSIBLE_BRANCH="${IMPOSSIBLE_BRANCH:-fast-sum-harness}"
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
    echo "corpus missing. From your laptop:"
    echo "  scp -P <port> -r datasets/qwen-emotion-stories root@<pod-ip>:/workspace/"
    exit 1
  fi
fi

uv sync
# uv.lock pins torch 2.14 built for CUDA 13, which needs a host driver that supports CUDA >= 13.0.
# RunPod: filter the deploy page by CUDA version 13.0+ (the template doesn't decide this; the host's driver does).
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
uv run python -c "
import torch
ok = torch.cuda.is_available()
print('torch', torch.__version__, 'built for CUDA', torch.version.cuda, '| GPU visible:', ok)
assert ok, 'torch cannot use the GPU: most likely the host driver is older than CUDA ' + str(torch.version.cuda) + '. Recreate the pod with the CUDA filter set to 13.0+.'
print('GPU:', torch.cuda.get_device_name(0))
"

# Everything below runs in the background under nohup. Three gates first; any failure stops the run before the sweep.
#   gate 1: the whole notebook (all 5 steps) at tiny settings on the GPU, with the smallest Coder
#   gate 2: the model-size comparison notebook, on gate 1's output
#   gate 3: GPU memory: the largest batch each size can generate at a full 15-attempt context
# Then the sweep: steps 1-4 for each size, smallest first, with gate 3's batch sizes.
# MODELS and STEP5 can be overridden, e.g. MODELS="Qwen/Qwen2.5-Coder-1.5B-Instruct" STEP5=1.
MODELS="${MODELS:-Qwen/Qwen2.5-Coder-0.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct}"
STEP5="${STEP5:-0}"     # 0: steps 1-4 per size (the size sweep); 1: also all 12 emotions
BASELINE_PROMPTS="${BASELINE_PROMPTS:-}"     # e.g. "NONE ANTI_HACK" for a shorter baseline
export MODELS STEP5 BASELINE_PROMPTS

nohup bash -c '
  set -o pipefail
  NB=src/experiments/20260919-emotion-steering-reward-hacking.ipynb
  CMP=src/experiments/20260919-steering-by-model-size.ipynb
  EXEC="uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1"
  fail() { echo "!!!!! $1. Not starting the sweep. Send this log to Claude."; exit 1; }

  echo "===== gate 1: whole notebook, tiny, on the GPU $(date)"
  RUN=laptop MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct STEP5=1 \
    $EXEC --output 20260919-steering.gate1.ipynb $NB || fail "gate 1 failed: see src/experiments/20260919-steering.gate1.ipynb"

  echo "===== gate 2: comparison notebook on the gate 1 output $(date)"
  RESULTS_DIR=steer-laptop $EXEC --output 20260919-steering-by-model-size.gate2.ipynb $CMP \
    || fail "gate 2 failed: see src/experiments/20260919-steering-by-model-size.gate2.ipynb"

  echo "===== gate 3: GPU memory, per model $(date)"
  uv run python src/scripts/memory_preflight.py $MODELS | tee /workspace/batch_sizes.txt || fail "gate 3 crashed"
  grep -q " 0$" /workspace/batch_sizes.txt && fail "gate 3: a model does not fit even one episode (batch 0 above)"

  echo "===== all gates passed; starting the sweep $(date)"
  for MODEL in $MODELS; do
    name=$(basename "$MODEL" | tr A-Z a-z)
    batch=$(grep "^$MODEL " /workspace/batch_sizes.txt | cut -d" " -f2)
    echo "===== $MODEL (gen_batch $batch) $(date)"
    RUN=runpod MODEL="$MODEL" STEP5=$STEP5 GEN_BATCH=$batch \
      $EXEC --output "20260919-steering.runpod.$name.ipynb" $NB \
      || echo "!!!!! $MODEL failed, continuing with the next size"
  done
  echo "===== all sizes done $(date)"
' > /workspace/run.log 2>&1 &
echo "started (pid $!). Follow it with: tail -f /workspace/run.log"
echo "The gates take ~20-30 min (mostly downloading 4 models). Results land in data/steer-runpod/<model>/."
echo "Copy them back with:  scp -P <port> -r root@<pod-ip>:/workspace/emotion-concepts/data/steer-runpod data/"
