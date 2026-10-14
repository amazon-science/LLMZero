# Setup: Conda (Local)

This guide covers running LLMZero with the **conda** compute backend. Training jobs run locally on the host machine inside a specified conda environment -- no Kubernetes or Docker required.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/)
- GPU drivers + CUDA toolkit (if using GPUs)

## 1. Create a Training Environment

Create a conda environment with your training framework:

```bash
conda create -n my_rl_env python=3.11 -y
conda activate my_rl_env

# Install the training framework. LLMZero defaults to verl (training_framework: verl_grpo).
pip install "verl==0.9.0"
# verl on PyPI may trail GitHub; if the pin is unavailable, install from source:
#   pip install "verl @ git+https://github.com/volcengine/verl@v0.9.0"
# Requires Python >=3.10 and CUDA-enabled PyTorch matching your GPUs.
# For a fully provisioned GPU image, see docker/ (verl 0.9.0 training environment).
```

## 2. Install LLMZero

In a separate environment or the same one:

```bash
git clone <repo-url> && cd LLMZero
pip install -e .
```

## 3. Run

```bash
llmzero \
    -i /path/to/data \
    -t "Your training instruction" \
    --compute-backend conda \
    --conda-env my_rl_env \
    --num-gpus 8
```

Or set defaults in `configs/default.yaml`:

```yaml
compute_infra:
  type: conda
  conda_env: "my_rl_env"
  num_gpus: 8
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--compute-backend conda` | Select the conda backend |
| `--conda-env NAME` | **(required)** Conda environment to activate for training |
| `--num-gpus N` | Number of GPUs available (informational, used in prompts) |

## How It Works

The conda backend:
1. Activates the specified conda environment via `conda shell.bash hook`
2. Runs the training script as a local subprocess
3. Streams stdout/stderr in real time (observers and early stopping work normally)
4. If the script was generated for Ray (`ray job submit` wrapper), the wrapper is automatically stripped

Job IDs are process PIDs. To stop a running job: `llmzero stop-job <PID> --compute-backend conda --conda-env my_rl_env`
