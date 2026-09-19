---
name: runpod
description: Manage RunPod GPU pods - setup SSH for Cursor/VS Code Remote SSH, sync files, list/start/stop pods. Use when the user asks to connect to RunPod, rent a GPU on RunPod, sync files to a RunPod pod, set up remote SSH for RunPod, or manage RunPod instances. Triggers on "runpod", "connect to my pod", "sync to pod", "GPU pod".
---

# RunPod Pod Management

## Configuration

**Before asking the user any setup questions**, use the values below as pre-filled answers.
Only ask when a value is still a placeholder (`<...>`), the request conflicts with one, or
the value is known-stale (pod IP/port always change on restart).

- **Use case**: emotion-vector steering runs of `src/experiments/20260919-emotion-steering-behaviours.ipynb` (and the size sweep before it), headless via `src/scripts/runpod_behaviours.sh`
- **Pod name**: `beh-<size>[-s<shard>]` (behaviours run) · `steer-coder-<size>` (size sweep)
- **Template**: `runpod-torch-v280` (PyTorch 2.8.0, CUDA 12.8.1, Ubuntu 24.04) — use official
  templates, not custom Docker images. Image pull takes 1-3 min on first boot per machine;
  this is unavoidable but the template is required for cu128 compatibility.
- **GPU**: 48 GB (L40S / A40 / RTX 6000 Ada) up to 7B; 80 GB (A100 SXM) for 14B — list options with `runpodctl gpu list`
- **Cloud type**: COMMUNITY first, fall back to SECURE (COMMUNITY consumer-GPU stock is
  almost always depleted)
- **Volume**: 50GB at `/workspace`
- **Container disk**: 20GB (template default)
- **Ports**: `8888/http,22/tcp`
- **SSH key**: `~/.ssh/id_rsa`
- **SSH host alias**: `runpod` (configured in `~/.ssh/config`)
- **Local project**: `/Users/bo/code/oss/emotion-concepts` (with `../impossible` as a sibling)
- **Remote path**: `/workspace/emotion-concepts` (and `/workspace/impossible`)
- **Dependencies**: managed via `pyproject.toml` / `uv.lock`

Keys live in the shell environment, never in this file:

```bash
export RUNPOD_API_KEY="..."
export HF_TOKEN="..."          # only if pulling gated Hugging Face models
```

## This project (emotion-concepts): read first

- **CUDA 13, not 12.8.** `uv.lock` pins torch 2.14 built for CUDA 13, so the *host driver* must support it:
  create pods with `--min-cuda-version 13.0`. The template's CUDA doesn't matter (uv brings its own libraries).
  Don't follow the cu128 re-locking advice below for this repo; it would change the laptop environment too.
- **Caches on the volume:** `runpod_behaviours.sh` sets `UV_CACHE_DIR=/workspace/.uv-cache` and
  `HF_HOME=/workspace/hf_cache`, since the 20 GB container disk can't hold them.
- **Corpus:** `datasets/qwen-emotion-stories` is gitignored and not on the Hub. `scp -r` it to `/workspace/` first.
- **Secure cloud** avoids the public-IP flags community cloud needs for direct SSH.
- **Delete pods** once results are fetched and verified locally (don't stop and resume; see the gotchas).

## Prerequisites

Install runpodctl if not present:
```bash
brew install runpod/runpodctl/runpodctl
```

API key: `RUNPOD_API_KEY` env var or run `runpodctl doctor` to configure.
Store key in environment, never hardcode.

## Workflow

### 0. Create a new pod

```bash
runpodctl pod create \
  --name "<pod-name>" \
  --gpu-id "<gpu-id from runpodctl gpu list>" \
  --template-id "runpod-torch-v280" \
  --volume-in-gb 50 \
  --volume-mount-path "/workspace" \
  --container-disk-in-gb 20 \
  --ports "8888/http,22/tcp" \
  --cloud-type COMMUNITY
```

If COMMUNITY fails with "does not have the resources", retry with `--cloud-type SECURE`.

Use `runpodctl gpu list` to see available GPU types and their IDs, and
`runpodctl template search torch` to find real template IDs.

### 1. List and identify pods

```bash
runpodctl pod list
```

### 2. Get pod SSH details

```bash
runpodctl pod get <pod-id> -o json
```

Look for the top-level `ssh` field — it contains `ip`, `port`, and `ssh_command`.
NOTE: `runtime` stays `null` on these template pods; don't poll it.

### 3. Configure SSH for Cursor/VS Code Remote SSH

**Critical**: Use the direct TCP connection (`ssh root@<ip> -p <port>`), NOT the
`ssh.runpod.io` proxy gateway. The proxy gateway does not support PTY, scp, rsync, or
VS Code/Cursor Remote SSH.

Add to `~/.ssh/config`:
```
Host runpod
    HostName <ip from pod get>
    Port <port from pod get>
    User root
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 15
    ServerAliveCountMax 8
    TCPKeepAlive yes
```

### 4. Test connection

```bash
ssh runpod "echo connected && hostname"
```

### 5. Sync files to pod

Use runpodctl send/receive for file transfer:

**From local:**
```bash
runpodctl send <path>
```
This outputs a receive code.

**On pod (via ssh):**
```bash
ssh runpod "cd /workspace && runpodctl receive <code>"
```

For ongoing sync after initial setup, rsync works via direct TCP:
```bash
rsync -avz --progress <local-path>/ runpod:/workspace/<remote-path>/
```

### 6. Pod management

```bash
runpodctl pod list          # List all pods
runpodctl pod stop <id>     # Stop (pause billing for GPU, storage still billed)
runpodctl pod start <id>    # Resume
runpodctl pod delete <id>   # Destroy (stops all billing)
```

## Post-Boot Setup

Run these steps after pod creation and SSH is ready:

1. **Symlink uv cache to workspace** (container disk is only 20GB, NVIDIA packages are ~19GB cached):
   ```bash
   ssh runpod 'rm -rf /root/.cache/uv && mkdir -p /workspace/.uv-cache && ln -s /workspace/.uv-cache /root/.cache/uv'
   ```

2. **Install uv**:
   ```bash
   ssh runpod 'pip install uv --break-system-packages'
   ```

3. **Sync project files** (use `--no-owner --no-group` to avoid chown errors):
   ```bash
   rsync -avz --no-owner --no-group --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude '*.pyc' <local-project>/ runpod:/workspace/<project>/
   ```

4. **Install dependencies** (this is the primary setup method — always `uv sync` on the pod
   rather than building custom Docker images):
   ```bash
   ssh runpod 'cd /workspace/<project> && uv lock && uv sync'
   ```

5. **Register Jupyter kernel** (so notebooks can find the venv):
   ```bash
   ssh runpod 'cd /workspace/<project> && .venv/bin/python -m ipykernel install --user --name <project> --display-name "<Project> (venv)"'
   ```

6. **Verify**:
   ```bash
   ssh runpod 'cd /workspace/<project> && .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
   ```

7. **Sync the lockfile back** after re-locking on the pod:
   ```bash
   rsync -avz --no-owner --no-group runpod:/workspace/<project>/uv.lock <local-project>/uv.lock
   ```

## CUDA / PyTorch Compatibility

- The template's NVIDIA driver supports CUDA 12.8 — torch must be built for cu128
- `pyproject.toml` uses `[[tool.uv.index]]` with `explicit = true` pointing to
  `https://download.pytorch.org/whl/cu128`
- Route all 15 `nvidia-*-cu12` packages to this index via `[tool.uv.sources]` to avoid
  PyPI version mismatches
- Do NOT use `uv pip install` to patch individual packages — it breaks the lockfile.
  Always edit `pyproject.toml` and re-run `uv lock && uv sync`

## Key gotchas

- `ssh.runpod.io` proxy = interactive SSH only. No scp, rsync, VS Code Remote.
  Direct TCP port (from `pod get` output) = full SSH. Use this for everything.
- **DON'T stop-and-resume — it reliably fails.** Stopped pods only resume on their original
  host, which is usually full by then ("not enough free GPUs on the host machine"). Prefer
  destroy + recreate fresh, or use a network volume / baked image to persist state.
- Pod persistent storage is at `/workspace` — files outside this are lost on restart, and
  the container disk is only 20GB. Redirect large caches onto the volume:
  `HF_HOME=/workspace/hf_cache`.
- Pod IP/port change on every restart — re-run `pod get` and update `~/.ssh/config`.
- SSH key must match what's in RunPod account settings (Settings > SSH Public Keys).
- If an RSA key gets "no mutual signature algorithm", add `PubkeyAcceptedAlgorithms +ssh-rsa`
  to the SSH config block.
- **`--env` vars set at create do NOT reach interactive SSH sessions.** Pass them inline:
  `HF_TOKEN=hf_... python3 ...`, or bake them into the kernel spec.
- **Deps on PEP 668 images**: `pip install --break-system-packages ...` (the image's Python is
  externally-managed; torch is preinstalled — don't reinstall it).
- **~1.7GB per-connection transfer cap**: rsync/scp of large files die mid-stream at a fixed
  byte offset (errors code 11/12). Workaround that works reliably:
  1. On pod: `split -b 300m file.pt xfer/file.part_`
  2. Pull each chunk over the SSH **exec** channel (not scp): `ssh runpod "cat /path/part" > local.part`
  3. Verify each chunk's byte size, then `cat local.part_* > file.pt` and check total vs a manifest.
- Run long jobs backgrounded with a log: `nohup python3 -u script.py > run.log 2>&1 &`
- **Don't hand-type image tags** — guessed tags fail and the pod exits instantly. Use
  `--template-id`, and find real ones via `runpodctl template search torch`.
- **GPU sizing**: roughly 2GB VRAM per 1B params at bf16, plus activations. A 13B model needs
  ≥40GB (A40/A6000-48GB/A100); a 24GB card (RTX 4090) tops out around 7-8B.
- **Cursor Remote-SSH**: Python + Jupyter extensions must be installed ON the remote, plus
  `ipywidgets` on the pod (else progress-bar render errors).
- **Gated model repos** (Llama and similar) need per-account approval on Hugging Face; an
  accepted licence on one account does not transfer to another.
