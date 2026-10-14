---
name: General - Download Model
description: "[General] Download a HuggingFace model to the model store, if not already present. Use before launching any training run that needs a local model. Framework-agnostic."
---

# Download Model

Download a HuggingFace model to the model storage directory, if not already present.

## Input

- **Model name**: HuggingFace model ID (e.g., `Qwen/Qwen3-4B`)

## Steps

### 1. Check if Model Exists

Read `constants/cluster.json` for `model_path_prefix` (default: `/fsx/hf`).

Construct the expected local path: `<model_path_prefix>/<model_name>` (e.g., `/fsx/hf/Qwen/Qwen3-4B`).

Check if the model directory exists and has content (e.g., contains `config.json`):

```bash
ls <model_path>/ 2>/dev/null | head -5
```

If the directory exists and has files, report "Model already downloaded" and skip.

### 2. Download the Model

The download method depends on `compute_backend` in `constants/cluster.json`:

**If `compute_backend == "raycluster"` (download on the remote cluster where `/fsx` is mounted):**

Construct the Ray address from `constants/cluster.json`:
```
http://<ray_service_name>.<ray_namespace>.svc.cluster.local:<ray_dashboard_port>
```

Submit a Ray job:
```bash
ray job submit \
  --address <ray_address> \
  --runtime-env-json '{"env_vars":{}}' \
  -- \
  bash -lc '
set -e
mkdir -p <model_path>
python3 - <<'"'"'PY'"'"'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="<model_name>",
    revision="main",
    local_dir="<model_path>",
    local_dir_use_symlinks=False
)
print("OK: cached at <model_path>")
PY
ls -al <model_path> | head
'
```

Monitor the job: `ray job status --address <ray_address> <job_id>`

**If `compute_backend == "conda"`, `"docker"`, or `"slurmdocker"` (download directly on the local/login machine):**

```bash
mkdir -p <model_path>
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='<model_name>',
    revision='main',
    local_dir='<model_path>',
    local_dir_use_symlinks=False,
)
print('OK: cached at <model_path>')
"
```

For docker backend: the model directory must be under one of the `docker_mount_dirs` (or `model_path_prefix`) so it's accessible inside the container.
For slurmdocker backend: the model directory must be on the shared filesystem (e.g. FSx) so compute nodes can access it.

### 3. Verify

Confirm the model directory now has the expected files:
```bash
ls <model_path>/config.json
```

### 4. Report

Tell the user:
- Whether the model was already present or freshly downloaded
- The local path to the model
- If download failed, the error message

## Important

- This step MUST complete before launching training
- Model download can take significant time for large models
- For raycluster, the download runs on the Ray cluster (where `/fsx` is mounted with fast I/O)
- For conda/docker/slurmdocker, the download runs locally — ensure the model path is on a filesystem with enough space
