#!/usr/bin/env bash
# Set up a RunPod GPU pod for src/experiments/20260919-emotion-steering-reward-hacking.ipynb, then run the
# model-size sweep: Qwen2.5-Coder 0.5B, 1.5B, 3B, 7B, one after another. Use a 48 GB GPU (L40S or A6000) for 7B.
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
uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"

# Headless, one model at a time, smallest first: every cell with RUN=runpod, saving each executed notebook.
# MODELS, STEP5 and GEN_BATCH can be overridden, e.g. MODELS="Qwen/Qwen2.5-Coder-1.5B-Instruct" STEP5=1.
MODELS="${MODELS:-Qwen/Qwen2.5-Coder-0.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-Coder-3B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct}"
STEP5="${STEP5:-0}"     # 0: steps 1-4 per size (the size sweep); 1: also all 12 emotions
NB=src/experiments/20260919-emotion-steering-reward-hacking.ipynb
nohup bash -c '
  # Smoke test first: tiny settings (RUN=laptop) on the GPU with the smallest Coder. Stop if anything breaks.
  echo "===== smoke test $(date)"
  RUN=laptop MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct STEP5=0 \
    uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output 20260919-steering.smoke.ipynb '"$NB"' \
    || { echo "!!!!! smoke test failed: see src/experiments/20260919-steering.smoke.ipynb. Not starting the sweep."; exit 1; }
  echo "===== smoke test passed $(date)"
  for MODEL in '"$MODELS"'; do
    name=$(basename "$MODEL" | tr A-Z a-z)
    batch=32; case "$name" in *7b*) batch=16;; esac        # 7B: smaller batches to fit long contexts in memory
    echo "===== $MODEL (gen_batch $batch) $(date)"
    RUN=runpod MODEL="$MODEL" STEP5='"$STEP5"' GEN_BATCH=$batch \
      uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 --output "20260919-steering.runpod.$name.ipynb" '"$NB"' \
      || echo "!!!!! $MODEL failed, continuing with the next size"
  done
  echo "===== all sizes done $(date)"
' > /workspace/run.log 2>&1 &
echo "started (pid $!). Follow it with: tail -f /workspace/run.log"
echo "Results land in data/steer-runpod/<model>/. Copy them back with:"
echo "  scp -P <port> -r root@<pod-ip>:/workspace/emotion-concepts/data/steer-runpod data/"
