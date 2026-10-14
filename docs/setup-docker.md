# Setup: Docker (Local)

This guide covers running LLMZero with the **Docker** compute backend. Training jobs run locally inside a Docker container -- no Kubernetes required.

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)
- GPU drivers on the host

## 1. Prepare a Training Image

Use any Docker image that contains your training framework:

```bash
# Pull a pre-built image
docker pull my-training:latest

# Or build your own
docker build -t my-training:latest .

# ECR images also work -- auto-login is handled automatically
# e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/repo:tag
```

To build the bundled **verl 0.9.0** training image (the default training
framework), use the provided Dockerfile:

```bash
./docker/build.sh    # builds llmzero-verl:0.9.0
```

See [docker/README.md](../docker/README.md) for build requirements and options.

## 2. Install LLMZero

On the host machine (not inside the container):

```bash
git clone <repo-url> && cd LLMZero
pip install -e .
```

## 3. Run

```bash
llmzero \
    -i /path/to/data \
    -t "Your training instruction" \
    --compute-backend docker \
    --docker-image my-training:latest \
    --mount-dir /path/to/data \
    --mount-dir /path/to/models \
    --num-gpus 8
```

Or set defaults in `configs/default.yaml`:

```yaml
compute_infra:
  type: docker
  image: "my-training:latest"
  gpu: "all"
  shm_size: "16g"
  mount_dirs: ["/path/to/data", "/path/to/models"]
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--compute-backend docker` | Select the Docker backend |
| `--docker-image IMAGE` | **(required)** Docker image (local, Docker Hub, or ECR URI) |
| `--mount-dir DIR` | Identity-mount a host directory (`-v /path:/path`); repeatable |
| `--num-gpus N` | Number of GPUs (informational, used in prompts) |

## Config-Only Options

These are available in `configs/default.yaml` but not exposed as CLI flags:

| Key | Default | Description |
|-----|---------|-------------|
| `gpu` | `"all"` | GPU device request (`--gpus` flag to `docker run`) |
| `shm_size` | `"16g"` | Shared memory size |
| `network` | `"host"` | Docker network mode |
| `volumes` | `[]` | Explicit `host:container` mounts (for non-identity mounts) |
| `extra_args` | `[]` | Extra arguments passed verbatim to `docker run` |
| `ecr_auto_login` | `true` | Auto-login to ECR if image is an ECR URI |

## How It Works

The Docker backend:
1. Builds a `docker run` command with GPU, volume, and network settings
2. If the image is from ECR, performs `aws ecr get-login-password` automatically
3. Runs the training script inside the container as a subprocess
4. Streams stdout/stderr in real time (observers and early stopping work normally)
5. If the script was generated for Ray (`ray job submit` wrapper), the wrapper is automatically stripped
6. Host directories passed via `--mount-dir` are identity-mounted so that absolute paths work unchanged inside the container

Job IDs are Docker container IDs. To stop a running job: `llmzero stop-job <container-id> --compute-backend docker --docker-image my-training:latest`
