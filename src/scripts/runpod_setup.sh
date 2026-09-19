#!/usr/bin/env bash
# Set up a RunPod GPU pod for src/experiments/20260919-emotion-steering-reward-hacking.ipynb, then start the run.
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

# Headless: runs every cell with RUN=runpod and saves the executed notebook, outputs included.
NB=src/experiments/20260919-emotion-steering-reward-hacking.ipynb
RUN=runpod nohup uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 --output 20260919-emotion-steering-reward-hacking.runpod.ipynb "$NB" \
  > /workspace/run.log 2>&1 &
echo "started (pid $!). Follow it with: tail -f /workspace/run.log"
echo "Results land in data/steer-runpod/. Copy them back with:"
echo "  scp -P <port> -r root@<pod-ip>:/workspace/emotion-concepts/data/steer-runpod data/"
