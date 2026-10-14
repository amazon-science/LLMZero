# Quick Start

Everything you need to go from a cloned repo to a running LLMZero search, an agentic test-time-scaling loop that spends inference-time compute to discover a better training strategy: install, the two ways to drive it (natural-language Agent Skills or the direct CLI), the compute backends, the training frameworks, and how configuration is layered.

## Install

```bash
git clone https://github.com/amazon-science/LLMZero && cd LLMZero
pip install -e .
```

Python 3.10 to 3.12.

## Option A: Claude Code Skills (natural language)

The lowest-friction way to drive LLMZero is to open Claude Code inside the cloned repo and describe your run in plain English. The LLMZero Agent Skills ship with the repo under `.claude/skills/`, so Claude Code auto-detects them and invokes the right ones for you. You do not type skill names or wire up a pipeline by hand: you say what you want, and the skills handle dataset prep, reward definition, task description, model download, launch, and reporting.

This works as long as the training framework and compute backend your run needs are already implemented. Out of the box that means verl 0.9.0 (`verl_grpo` and `verl_sft`) on the `raycluster`, `conda`, `docker`, `slurm`, and `slurmdocker` backends. If you need something new, add it first (see [Training Frameworks](#training-frameworks) and [Compute Backends](#compute-backends)) and the skills will then pick it up.

First, set your compute backend and defaults in `constants/cluster.json` (see [Configuration](#configuration)). Then just describe the run. For example, in Claude Code inside the repo:

> Run a GRPO post-training search on `<hf-org>/<dataset>` with `Qwen/Qwen3-4B`, 16 search iterations, on my Slurm cluster: partition `<partition>`, 1 node of 8 GPUs, using the verl 0.9.0 enroot image at `/path/to/llmzero-verl-0.9.0.sqsh`. Cap val and test at 500 rows each, then launch and give me a report.

Claude Code recognizes this as the end-to-end orchestrator and runs it:

```
/verl-grpo-run-training
```

Or invoke each step on its own (the orchestrator just chains these):

```
/verl-grpo-prepare-dataset   <hf-org>/<dataset>   # download + convert to verl parquet
/verl-grpo-define-reward      <run_folder>        # write a compute_score reward
/verl-grpo-generate-task      <run_folder>        # render description.txt
/general-download-model       Qwen/Qwen3-4B       # fetch the model
/verl-grpo-validate-run       <run_folder>        # health-check before launch
/general-launch-llmzero       <run_folder>        # run the MCTS search
/general-check-training       <run_folder>        # monitor progress
/general-gather-results       <run_folder>        # parse output, write a report
```

The full set of skills:

| Skill | Purpose |
|-------|---------|
| `/verl-grpo-run-training` | End-to-end orchestrator: dataset to report |
| `/verl-grpo-prepare-dataset` | Download an HF dataset, convert to verl parquet format |
| `/verl-grpo-define-reward` | Write a `compute_score` reward function |
| `/verl-grpo-generate-task` | Create `description.txt` from a template + run config |
| `/verl-grpo-validate-run` | Health-check a run folder before launch |
| `/general-download-model` | Download an HF model locally |
| `/general-launch-llmzero` | Invoke the `llmzero` CLI for MCTS search |
| `/general-check-training` | Check job status and search progress |
| `/general-gather-results` | Parse output and write a report |
| `/general-plot-pareto` | Plot the score progression across MCTS nodes |
| `/general-add-training-framework` | Scaffold a new training-framework plugin |

The directory-name prefix encodes the category: `general-*` skills are framework-agnostic, `verl-grpo-*` skills are RL post-training. The invocation name is the directory name.

## Option B: Direct CLI

If you prefer to drive the engine yourself, call the `llmzero` CLI. The required arguments are an input folder (`-i`, which must contain only `description.txt`), an output folder (`-o`), a config file (`-c`), the number of search iterations (`-n`), and a compute backend with its flags. The examples below cover all four runnable backends.

```bash
# Conda: local machine, conda env
llmzero -i ./my_task -t "Train Qwen3-4B with 8 A100 GPUs" -n 16 \
    --compute-backend conda --conda-env my_rl_env --num-gpus 8

# Docker: local machine, container
llmzero -i ./my_task -t "Train Qwen3-4B with 8 A100 GPUs" -n 16 \
    --compute-backend docker --docker-image my-training:latest \
    --mount-dir /data --mount-dir /models --num-gpus 8

# Slurm + Docker: remote Slurm nodes, pyxis/enroot
llmzero -i ./my_task -t "Train Qwen3-4B with 8 A100 GPUs" -n 16 \
    --compute-backend slurmdocker --slurm-partition <partition> \
    --docker-image /path/to/llmzero-verl-0.9.0.sqsh \
    --mount-dir /fsx:/fsx --slurm-nodes 4 --slurm-gpus-per-node 8

# Ray on Kubernetes (EKS): requires a deployed Ray cluster + agent pod
llmzero -i ./my_task -t "Train Qwen3-4B with 8 A100 GPUs" -n 16 \
    --compute-backend raycluster --ray-service my-cluster-head-svc \
    --ray-namespace my-namespace --ray-port 8265 --entrypoint-num-gpus 8
```

A fuller launch, pointing at an assembled run folder (the layout the Agent Skills produce), looks like this:

```bash
llmzero -i <run_folder>/llmzero_input \
        -o <run_folder>/llmzero_output \
        -c <run_folder>/llmzero_config.yaml \
        --compute-backend slurmdocker \
        --slurm-partition <partition> \
        --docker-image /path/to/llmzero-verl-0.9.0.sqsh \
        --mount-dir /fsx:/fsx \
        --slurm-nodes 1 --slurm-gpus-per-node 8 \
        -n 16 -v 2
```

Each `--mount-dir` uses `src:dst` format (identity-mount a host directory into the container at the same path so absolute host paths work unchanged). Add `--slurm-account <account>` and `--slurm-time <limit>` for `slurmdocker` if your cluster needs them. Useful extra flags: `--stop-after-success` (stop at the first successful run instead of continuing to optimize) and `-j <N>` / `--job-parallel <N>` (run N training jobs concurrently via async Top-K weighted deepening instead of sequential MCTS).

Two companion commands manage individual jobs, using the same backend flags:

```bash
llmzero submit-script <script.sh> [backend flags]   # run a script, stream its output, return a job ID
llmzero stop-job <job-id> [backend flags]           # terminate a running job
```

Run `llmzero --help` for the full flag reference.

## Compute Backends

Each training run in the strategy search executes on a compute backend. Five pluggable backends live behind the `BaseComputeBackend` ABC in `src/autogluon/llmzero/compute_infra/`. The agent (the search loop) and the training jobs can run in different places:

| Backend | Training runs on | Agent runs on |
|---------|------------------|---------------|
| `raycluster` | Remote Ray cluster on Kubernetes (EKS/HyperPod) | K8s agent pod |
| `conda` | Local machine (conda env) | Local |
| `docker` | Local machine (container) | Local |
| `slurm` | Remote Slurm nodes (conda env, multi-node Ray) | Login node |
| `slurmdocker` | Remote Slurm nodes (containerized via pyxis/enroot) | Login node |

Each backend implements `prepare_script()`, `submit_job()`, `terminate()`, and `extract_job_id()`. The submission command blocks and streams the remote job's stdout line by line so the execution observers can parse validation scores, checkpoint paths, and step metrics in real time.

**Docker `mount_dirs`:** identity-mount host directories into the container at the same path (for example `/fsx:/fsx`) so all absolute host paths work unchanged inside the container. Mount the runs root, the model prefix, and any data directories.

**Logging note (`slurmdocker`, `conda`, `docker`):** on these backends verl spawns Ray workers internally, and environment variables (including `WANDB_API_KEY`) do not propagate to them. Use `trainer.logger=["console"]` (or add `"tensorboard"`); `"wandb"` crashes immediately with `No API key configured`. Only the `raycluster` backend (via `ray job submit --runtime-env-json`) propagates env vars to workers correctly.

## Training Frameworks

A training framework is the actual trainer each strategy runs; it also tells the search agents what to optimize. Training frameworks are a pluggable registry behind the `BaseTrainingFramework` ABC in `src/autogluon/llmzero/training_frameworks/`. `verl_grpo` (verl 0.9.0 GRPO/PPO) is the default; `verl_sft` (verl 0.9.0 supervised fine-tuning) ships too. The framework is selected via `training_framework` in config (default: `verl_grpo`).

Each framework declares:

- **Hyperparameter categories and descriptions** that guide the proposer when it proposes the next node's strategy (the hyperparameters are part of that strategy).
- **Metric descriptions** so the agents understand what each logged value means.
- **Default text metrics and plot metrics** (what to track and chart across nodes).
- **Checkpoint-resume logic** for warm-starting a strong node from its parent's checkpoint instead of retraining from scratch. `verl_grpo` supports verl 0.9.0 native checkpoint save/resume (including LoRA-only) for continuous training; `verl_sft` ships single-shot by default (every node trains its own config from the base model).
- **The subprocess stream to parse** via `log_stream()`: `"stdout"` (default), `"stderr"`, or `"both"`, so observers ingest metrics from the correct stream (verl 0.9.0 GRPO, for example, logs checkpoint-save lines to stderr while metrics go to stdout, so `verl_grpo` declares `"both"`).

The base class returns empty defaults, so an unset config means no framework-specific guidance. `BaseAgent` and `BasePrompt` expose a `self.training_framework` property for convenient access.

**Adding one:** subclass `BaseTrainingFramework` in `training_frameworks/<name>.py`, implement the required methods, and register it via a module import in `__init__.py`. The `/general-add-training-framework` skill scaffolds a new plugin from a real training log. See [docs/onboarding-a-training-framework.md](onboarding-a-training-framework.md) for a worked non-RFT onboarding and a generic checklist.

**LLM providers for the search agents:** the team of specialized LLM agents that drives the search (task/data perception, tool selection, retrieval, code generation, execution, error analysis, reranking, early stopping, proposing) runs on any of Amazon Bedrock, OpenAI, Anthropic, or SageMaker. Per-agent model, temperature, max tokens, and thinking configuration live in the Hydra config (see [Configuration](#configuration)).

## Configuration

Configuration is layered in three places; CLI flags always win at runtime.

**`constants/cluster.json`** sets the compute backend, hardware, and model defaults used by the Agent Skills: `compute_backend`, backend-specific fields (partition / docker image / mounts / ray service, etc.), `nnodes`, `n_gpus_per_node`, `model_path_prefix`, `default_model`, `runs_root`, `verl_output_dir`, `llmzero_path`, the HF access token, the WandB API key, and `job_parallel`. Per-backend starting points are provided as `constants/cluster.<backend>.example.json`:

```
constants/cluster.conda.example.json
constants/cluster.docker.example.json
constants/cluster.raycluster.example.json
constants/cluster.slurmdocker.example.json
```

Copy the one for your backend to `constants/cluster.json` and fill in the real values (replace the `/path/to/...` and empty-string placeholders with your partition, container image, HF token, and so on).

**Hydra YAML under `src/autogluon/llmzero/configs/`** (`default.yaml`) holds the search-loop parameters and agent settings:

- **MCTS / search loop:** `exploration_constant` (UCT exploration weight), `max_debug_depth` (debug attempts before abandoning a node), and the rest of the tree-search knobs.
- **Per-agent LLM settings:** model, temperature, max tokens, and thinking config for each of the LLM agents, plus the provider (Bedrock, OpenAI, Anthropic, SageMaker). Provider-specific presets ship as `bedrock.yaml` and `openai.yaml` alongside `default.yaml`.
- **Early stopper and proposer** settings.
- **Scoring:** `validation_metric_pattern` (the regex the observers use to extract the target metric from the training logs; required for scoring), `training_framework`, and `compute_infra`.

CLI flags override config values at runtime, so you can keep a stable `llmzero_config.yaml` per run and tweak backend, iteration count, or parallelism from the command line.

## Examples

Two self-contained, reproducible example run folders, each a complete strategy search you can rerun end to end, ship under [examples/](../examples/):

- [examples/grpo_ssmr_qwen3_4b/](../examples/grpo_ssmr_qwen3_4b/) : GRPO LoRA post-training of `Qwen/Qwen3-4B` on SSMR-Bench (music-theory multiple choice), with checkpoint chaining (warm-start deepening) enabled.
- [examples/sft_govreport_qwen3_4b/](../examples/sft_govreport_qwen3_4b/) : verl SFT of `Qwen/Qwen3-4B` on long-context GovReport summarization, single-shot search minimizing `val/loss`.

Each folder's `README.md` walks through regenerating the data, assembling the run folder, and launching via both the skills pipeline and the direct CLI. Full measured results and method for both benchmarks are in [docs/benchmarks.md](benchmarks.md).
