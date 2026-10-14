# Docker: verl 0.9.0 Training Image

This directory builds the GPU training image LLMZero submits jobs into. The
image contains the **verl 0.9.0** training stack (PyTorch, vLLM, Megatron,
TransformerEngine, flash-attn, apex). LLMZero itself — the orchestration
agent — runs **outside** the container and submits training jobs into it through
the `docker`, `slurmdocker`, or `raycluster` compute backends.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.verl` | verl 0.9.0 training environment (thin layer over verl's prebuilt stable image, pinned to `v0.9.0`) |
| `build.sh` | Convenience wrapper around `docker build` |

## Provenance

`Dockerfile.verl` is a thin layer on top of verl's official prebuilt stable image
[`verlai/verl:vllm024.dev2`](https://hub.docker.com/r/verlai/verl/tags), which bakes
verl 0.9.0's full dependency closure (vLLM 0.24.0 / torch 2.11.0 / CUDA 13.0.2 /
Ubuntu 24.04 / Python 3.12 / transformers 5.x). It is NOT built from source.

> **Note on the base tag.** verl's README advertises `verlai/verl:vllm024.latest`,
> but as of this writing that tag is not yet published on Docker Hub — the
> Dockerfile pins `verlai/verl:vllm024.dev2` (the current published 0.24.0 prebuilt).
> Match the base tag to the verl version: verl 0.9.0 requires `vllm>=0.18.0` and
> ships against vLLM 0.24.0. Do NOT pair verl 0.9.0 with the older `vllm017` base.

The prebuilt `verlai/verl` images bake verl's dependency closure but uninstall the
`verl` package itself (so a developer can mount their own checkout). This image
**installs verl 0.9.0 on top** (`pip install --no-deps git+…@v0.9.0`) so it is a
ready-to-run training environment. To develop against a local verl checkout instead,
mount it and reinstall: `-v /path/to/verl:/verl` then `pip install -e /verl`.

## Build

The base image is pulled from Docker Hub and this Dockerfile adds a single
`pip install --no-deps verl@v0.9.0` layer, so the build is fast (no from-source
compilation). The pulled base image is large (~12 GB).

```bash
# From the repo root
./docker/build.sh                       # builds llmzero-verl:0.9.0
./docker/build.sh my-registry/llmzero-verl:0.9.0   # custom tag

# Optional: pass extra docker build args (e.g. an APT mirror)
./docker/build.sh llmzero-verl:0.9.0 --build-arg APT_MIRROR=http://mirror.example.com/ubuntu
```

Pin a different verl version at build time with `--build-arg VERL_VERSION=v0.9.0`
(and match `--build-arg BASE_IMAGE=` to that verl version's blessed vLLM base).

## Use with LLMZero

### Local Docker backend

```bash
llmzero \
    -i ./my_task \
    -t "Train Qwen3-4B with GRPO" \
    --compute-backend docker \
    --docker-image llmzero-verl:0.9.0 \
    --mount-dir /data \
    --num-gpus 8
```

### Slurm + Docker (pyxis/enroot)

Convert the image to a `.sqsh` with enroot, then:

```bash
llmzero \
    -i ./my_task \
    -t "Train Qwen3-4B with GRPO" \
    --compute-backend slurmdocker \
    --slurm-partition gpu \
    --docker-image /path/to/images/llmzero-verl-0.9.0.sqsh \
    --slurm-nodes 1 \
    --slurm-gpus-per-node 8
```

### Ray on Kubernetes

Push the image to your registry and use it as the worker container image when
deploying the Ray cluster (see [../docs/setup-raycluster.md](../docs/setup-raycluster.md)).

## Push to a registry

```bash
docker tag llmzero-verl:0.9.0 <your-registry>/llmzero-verl:0.9.0
docker push <your-registry>/llmzero-verl:0.9.0
```
