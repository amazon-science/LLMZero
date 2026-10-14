# LLMZero CLI - Ray Job Management Commands

Quick reference for the new Ray job management CLI commands.

**Note:** The main agent command still works as before: `llmzero -i <input> [options]`
The new commands are subcommands that extend the CLI functionality.

## New Commands

### 1. submit-script

Submit a bash script as a Ray job.

```bash
llmzero submit-script <script> -r <ray-service> [options]
```

**Example:**
```bash
llmzero submit-script train.sh -r my-ray-head-svc
```

**Returns:**
- Exit code: 0 (success) or 1 (failure)
- Stdout: Job ID on last line (e.g., `raysubmit_abc123`)

### 2. stop-job

Stop a running Ray job by ID.

```bash
llmzero stop-job <job-id> -r <ray-service> [options]
```

**Example:**
```bash
llmzero stop-job raysubmit_abc123 -r my-ray-head-svc
```

**Returns:**
- Exit code: 0 (success) or 1 (failure)

## Common Options

Both commands accept:

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --ray-service` | Ray head service name (required) | - |
| `--ray-namespace` | Kubernetes namespace | `llmzero` |
| `--ray-port` | Ray dashboard port | `8265` |
| `-v, --verbosity` | Logging level (0-4) | `2` |

## Quick Examples

### Submit and Capture Job ID

```bash
JOB_ID=$(llmzero submit-script train.sh -r my-ray-head-svc 2>/dev/null | tail -1)
echo "Job ID: $JOB_ID"
```

### Submit and Monitor

```bash
# Submit
JOB_ID=$(llmzero submit-script train.sh -r my-ray-head-svc 2>/dev/null | tail -1)

# Monitor (using Ray CLI)
RAY_ADDR="http://my-ray-head-svc.llmzero.svc.cluster.local:8265"
ray job status $JOB_ID --address $RAY_ADDR

# Stop if needed
llmzero stop-job $JOB_ID -r my-ray-head-svc
```

### Submit with Custom Namespace

```bash
llmzero submit-script train.sh \
  -r my-cluster-head-svc \
  --ray-namespace my-namespace \
  --ray-port 8265
```

## Script Format

Your bash script should contain Ray job submission commands:

```bash
#!/bin/bash
ray job submit \
    --address http://localhost:8265 \
    --working-dir /path/to/training \
    -- python train.py --epochs 10
```

**Note:** The `--address http://localhost:8265` will be automatically rewritten to use the correct Kubernetes service DNS.

## Testing

### Quick Test

```bash
# 1. Create test script
cat > test.sh <<'EOF'
#!/bin/bash
ray job submit \
    --address http://localhost:8265 \
    --working-dir /tmp \
    -- python -c "import ray; ray.init(); print('Hello Ray!')"
EOF

# 2. Submit it
llmzero submit-script test.sh -r my-ray-head-svc

# 3. Should output:
# Submitting script: test.sh
# ✓ Successfully submitted Ray job: raysubmit_xyz
# Job ID: raysubmit_xyz
# raysubmit_xyz
```

## Help Commands

```bash
llmzero submit-script --help
llmzero stop-job --help
```

## Code Location

- **Executor:** `src/autogluon/llmzero/execution/executor.py`
  - Methods: `submit_ray_script()`, `stop_ray_job()`
- **CLI:** `src/autogluon/llmzero/cli/app.py`
  - Commands: `submit-script`, `stop-job`

## Requirements

- LLMZero installed
- Ray cluster running on Kubernetes
- Access to Ray head service
- Bash scripts with Ray job submission commands
