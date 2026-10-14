#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing.resource_tracker
from pathlib import Path

import typer

from autogluon.llmzero.coding_agent import run_agent
from autogluon.llmzero.constants import DEFAULT_CONFIG_PATH


def _noop(*args, **kwargs):
    pass


multiprocessing.resource_tracker.register = _noop
multiprocessing.resource_tracker.unregister = _noop
multiprocessing.resource_tracker.ensure_running = _noop


app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # === Run parameters ===
    input_data_folder: str = typer.Option(None, "-i", "--input", help="Path to data folder"),
    output_dir: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory (if omitted, auto-generated under runs/)",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "-c",
        "--config",
        help=f"YAML config file (default: {DEFAULT_CONFIG_PATH})",
    ),
    llm_provider: str = typer.Option(
        "bedrock",
        "--provider",
        help="LLM provider to use (bedrock, openai, anthropic, sagemaker). Overrides config file.",
    ),
    max_iterations: int = typer.Option(
        16,
        "-n",
        "--max-iterations",
        help="Max iteration count. If the task hasn't succeeded after this many iterations, it will terminate.",
    ),
    stop_after_success: bool = typer.Option(
        False,
        "--stop-after-success",
        help="If enabled, the system will stop after finding the first valid solution instead of continuing to optimize. By default, the system keeps searching for better solutions until reaching the maximum number of iterations.",
    ),
    enable_per_iteration_instruction: bool = typer.Option(
        False,
        "--enable-per-iteration-instruction",
        help="If enabled, provide an instruction at the start of each iteration (except the first, which uses the initial instruction). The process suspends until you provide it.",
    ),
    initial_user_input: str | None = typer.Option(
        None, "-t", "--initial-instruction", help="You can provide the initial instruction here."
    ),
    # === Compute Infrastructure Configuration ===
    compute_backend: str | None = typer.Option(
        None,
        "--compute-backend",
        help="Compute backend type: 'raycluster', 'conda', 'docker', 'slurm', or 'slurmdocker' (overrides config file)",
    ),
    # --- Ray Cluster options ---
    ray_namespace: str | None = typer.Option(
        "llmzero",
        "--ray-namespace",
        help="Kubernetes namespace for Ray cluster (overrides config file)",
    ),
    ray_service_name: str | None = typer.Option(
        None,
        "-r",
        "--ray-service",
        help="Ray cluster head service name (overrides config file)",
    ),
    ray_dashboard_port: int | None = typer.Option(
        None,
        "--ray-port",
        help="Ray dashboard port (overrides config file, default: 8265)",
    ),
    entrypoint_num_gpus: int | None = typer.Option(
        None,
        "--entrypoint-num-gpus",
        help="Number of GPUs for ray job submit entrypoint (routes job to GPU workers)",
    ),
    # --- Conda backend options ---
    conda_env: str | None = typer.Option(
        None,
        "--conda-env",
        help="Conda environment name for conda backend (overrides config file)",
    ),
    num_gpus: int | None = typer.Option(
        None,
        "--num-gpus",
        help="Number of GPUs for conda/docker backend (overrides config file)",
    ),
    # --- Docker backend options ---
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Docker image for docker backend (overrides config file)",
    ),
    mount_dirs: list[str] | None = typer.Option(
        None,
        "--mount-dir",
        help="Host directory to identity-mount into docker container (repeatable, e.g. --mount-dir /data --mount-dir /models)",
    ),
    # --- Slurm+Docker backend options (slurmdocker) ---
    slurm_partition: str | None = typer.Option(
        None,
        "--slurm-partition",
        help="Slurm partition name for slurmdocker backend (e.g. 'gpu')",
    ),
    slurm_num_nodes: int | None = typer.Option(
        None,
        "--slurm-nodes",
        help="Number of Slurm nodes (overrides config file)",
    ),
    slurm_gpus_per_node: int | None = typer.Option(
        None,
        "--slurm-gpus-per-node",
        help="GPUs per node for slurmdocker backend (overrides config file)",
    ),
    slurm_time_limit: str | None = typer.Option(
        None,
        "--slurm-time",
        help="Slurm time limit, e.g. '1-00:00:00' (overrides config file)",
    ),
    slurm_account: str | None = typer.Option(
        None,
        "--slurm-account",
        help="Slurm account name (overrides config file)",
    ),
    # === WandB Configuration ===
    wandb_api_key: str | None = typer.Option(
        None,
        "--wandb-api-key",
        help="Weights & Biases API key for experiment tracking. If provided, wandb logging will be enabled in generated training scripts.",
    ),
    wandb_project_postfix: str = typer.Option(
        "",
        "--project-postfix",
        help="Optional postfix appended to wandb project names (e.g., '_v2', '_debug'). Helps organize experiments by variant.",
    ),
    # === Run Configuration ===
    run_postfix: str = typer.Option(
        "",
        "--run-postfix",
        help="Optional postfix appended to node output folder names (e.g., '_enable_early_stop').",
    ),
    job_parallel: int = typer.Option(
        1,
        "-j",
        "--job-parallel",
        help="Number of parallel training jobs. If >1, uses async search with Top-K Weighted Deepening instead of sequential MCTS.",
    ),
    validation_score_mode: str | None = typer.Option(
        None,
        "--validation-score-mode",
        help=(
            "How a run's validation metric becomes the MCTS node score: "
            "'best_validation' (direction-aware best across the run) or "
            "'final_validation' (last recorded value). Overrides config."
        ),
    ),
    # === Logging parameters ===
    verbosity: int = typer.Option(
        1,
        "-v",
        "--verbosity",
        help=(
            "-v 0: Only includes error messages\n"
            "-v 1: Contains key essential information\n"
            "-v 2: Includes brief information plus detailed information such as file save locations\n"
            "-v 3: Includes info-level information plus all model training related information\n"
            "-v 4: Includes full debug information"
        ),
    ),
):
    """
    llmzero: a CLI for LLM Post Training.
    """
    # If a subcommand was invoked, don't run the agent
    if ctx.invoked_subcommand is not None:
        return

    # Validate required options for agent run
    if input_data_folder is None:
        typer.echo("Error: Missing option '-i' / '--input'", err=True)
        typer.echo("Run 'llmzero --help' for usage information", err=True)
        raise typer.Exit(code=1)

    # Invoke the core run_agent function
    # Override config path if provider is specified and config path is default
    provider_config_path = config_path
    if llm_provider in ["bedrock", "openai", "anthropic", "sagemaker"] and config_path == DEFAULT_CONFIG_PATH:
        provider_config_path = Path(DEFAULT_CONFIG_PATH).parent / f"{llm_provider}.yaml"
        if not provider_config_path.exists():
            provider_config_path = DEFAULT_CONFIG_PATH

    # Build compute infra overrides from CLI flags
    compute_infra_overrides = {}
    if compute_backend is not None:
        compute_infra_overrides["type"] = compute_backend
    # Ray cluster overrides
    if ray_namespace is not None:
        compute_infra_overrides["kubernetes_namespace"] = ray_namespace
    if ray_service_name is not None:
        compute_infra_overrides["ray_service_name"] = ray_service_name
    if ray_dashboard_port is not None:
        compute_infra_overrides["ray_dashboard_port"] = ray_dashboard_port
    if entrypoint_num_gpus is not None:
        compute_infra_overrides["entrypoint_num_gpus"] = entrypoint_num_gpus
    # Conda backend overrides
    if conda_env is not None:
        compute_infra_overrides["conda_env"] = conda_env
    if num_gpus is not None:
        compute_infra_overrides["num_gpus"] = num_gpus
    # Docker / slurmdocker backend overrides. The two backends name these keys
    # differently: docker expects image/mount_dirs, slurmdocker expects
    # container_image/container_mounts (see each backend's validate_config). Map
    # the shared --docker-image/--mount-dir flags to the right keys per backend.
    if compute_backend == "slurmdocker":
        if docker_image is not None:
            compute_infra_overrides["container_image"] = docker_image
        if mount_dirs is not None:
            compute_infra_overrides["container_mounts"] = mount_dirs
    else:
        if docker_image is not None:
            compute_infra_overrides["image"] = docker_image
        if mount_dirs is not None:
            compute_infra_overrides["mount_dirs"] = mount_dirs
    # Slurm backend overrides
    if slurm_partition is not None:
        compute_infra_overrides["partition"] = slurm_partition
    if slurm_num_nodes is not None:
        compute_infra_overrides["num_nodes"] = slurm_num_nodes
    if slurm_gpus_per_node is not None:
        compute_infra_overrides["gpus_per_node"] = slurm_gpus_per_node
    if slurm_time_limit is not None:
        compute_infra_overrides["time_limit"] = slurm_time_limit
    if slurm_account is not None:
        compute_infra_overrides["account"] = slurm_account
    # conda_env is shared between conda and slurmdocker backends (already mapped above)

    run_agent(
        input_data_folder=input_data_folder,
        output_folder=output_dir,
        config_path=str(provider_config_path),
        max_iterations=max_iterations,
        stop_after_success=stop_after_success,
        enable_per_iteration_instruction=enable_per_iteration_instruction,
        initial_user_input=initial_user_input,
        compute_infra_overrides=compute_infra_overrides if compute_infra_overrides else None,
        wandb_api_key=wandb_api_key,
        wandb_project_postfix=wandb_project_postfix,
        run_postfix=run_postfix,
        job_parallel=job_parallel,
        validation_score_mode=validation_score_mode,
        verbosity=verbosity,
    )


def _build_compute_backend_config(
    backend_type: str,
    *,
    ray_namespace: str = "llmzero",
    ray_service_name: str | None = None,
    ray_dashboard_port: int = 8265,
    conda_env: str | None = None,
    num_gpus: int = 0,
    docker_image: str | None = None,
    mount_dirs: list[str] | None = None,
    slurm_partition: str | None = None,
    slurm_num_nodes: int | None = None,
    slurm_gpus_per_node: int | None = None,
    slurm_time_limit: str | None = None,
    slurm_account: str | None = None,
    no_container_mount_home: bool = False,
    container_env: list[str] | None = None,
) -> dict:
    """Build a compute backend config dict from CLI arguments, with validation."""
    if backend_type == "conda":
        if conda_env is None:
            typer.echo("Error: --conda-env is required for conda backend", err=True)
            raise typer.Exit(code=1)
        return {"type": "conda", "conda_env": conda_env, "num_gpus": num_gpus}
    elif backend_type == "docker":
        if docker_image is None:
            typer.echo("Error: --docker-image is required for docker backend", err=True)
            raise typer.Exit(code=1)
        config: dict = {"type": "docker", "image": docker_image}
        if mount_dirs:
            config["mount_dirs"] = mount_dirs
        if num_gpus:
            config["num_gpus"] = num_gpus
        return config
    elif backend_type == "slurm":
        if slurm_partition is None:
            typer.echo("Error: --slurm-partition is required for slurm backend", err=True)
            raise typer.Exit(code=1)
        config = {"type": "slurm", "partition": slurm_partition}
        if conda_env is not None:
            config["conda_env"] = conda_env
        if slurm_num_nodes is not None:
            config["num_nodes"] = slurm_num_nodes
        if slurm_gpus_per_node is not None:
            config["gpus_per_node"] = slurm_gpus_per_node
        if slurm_time_limit is not None:
            config["time_limit"] = slurm_time_limit
        if slurm_account is not None:
            config["account"] = slurm_account
        return config
    elif backend_type == "slurmdocker":
        if slurm_partition is None:
            typer.echo("Error: --slurm-partition is required for slurmdocker backend", err=True)
            raise typer.Exit(code=1)
        if docker_image is None:
            typer.echo("Error: --docker-image is required for slurmdocker backend", err=True)
            raise typer.Exit(code=1)
        config = {
            "type": "slurmdocker",
            "partition": slurm_partition,
            "container_image": docker_image,
        }
        if mount_dirs:
            config["container_mounts"] = mount_dirs
        if slurm_num_nodes is not None:
            config["num_nodes"] = slurm_num_nodes
        if slurm_gpus_per_node is not None:
            config["gpus_per_node"] = slurm_gpus_per_node
        if slurm_time_limit is not None:
            config["time_limit"] = slurm_time_limit
        if slurm_account is not None:
            config["account"] = slurm_account
        if no_container_mount_home:
            config["no_container_mount_home"] = True
        if container_env:
            env_dict = {}
            for item in container_env:
                key, _, val = item.partition("=")
                env_dict[key] = val
            config["container_env"] = env_dict
        return config
    else:
        if ray_service_name is None:
            typer.echo("Error: --ray-service / -r is required for raycluster backend", err=True)
            raise typer.Exit(code=1)
        return {
            "type": "raycluster",
            "kubernetes_namespace": ray_namespace,
            "ray_service_name": ray_service_name,
            "ray_dashboard_port": ray_dashboard_port,
        }


@app.command(name="submit-script")
def submit_script(
    script_path: Path = typer.Argument(..., help="Path to bash script to submit as a compute job"),
    compute_backend: str = typer.Option(
        "raycluster",
        "--compute-backend",
        help="Compute backend: 'raycluster', 'conda', 'docker', 'slurm', or 'slurmdocker'",
    ),
    # --- Ray Cluster options ---
    ray_namespace: str = typer.Option(
        "llmzero",
        "--ray-namespace",
        help="Kubernetes namespace for Ray cluster",
    ),
    ray_service_name: str | None = typer.Option(
        None,
        "-r",
        "--ray-service",
        help="Ray cluster head service name",
    ),
    ray_dashboard_port: int = typer.Option(
        8265,
        "--ray-port",
        help="Ray dashboard port",
    ),
    # --- Conda backend options ---
    conda_env: str | None = typer.Option(
        None,
        "--conda-env",
        help="Conda environment name for conda backend",
    ),
    num_gpus: int = typer.Option(
        0,
        "--num-gpus",
        help="Number of GPUs for conda/docker backend",
    ),
    # --- Docker backend options ---
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Docker image for docker backend",
    ),
    mount_dirs: list[str] | None = typer.Option(
        None,
        "--mount-dir",
        help="Host directory to identity-mount into docker container (repeatable, e.g. --mount-dir /data --mount-dir /models)",
    ),
    # --- Slurm backend options ---
    slurm_partition: str | None = typer.Option(
        None,
        "--slurm-partition",
        help="Slurm partition name for slurmdocker backend (e.g. 'gpu')",
    ),
    slurm_num_nodes: int | None = typer.Option(
        None,
        "--slurm-nodes",
        help="Number of Slurm nodes",
    ),
    slurm_gpus_per_node: int | None = typer.Option(
        None,
        "--slurm-gpus-per-node",
        help="GPUs per node for slurm backend",
    ),
    slurm_time_limit: str | None = typer.Option(
        None,
        "--slurm-time",
        help="Slurm time limit, e.g. '30-00:00:00'",
    ),
    slurm_account: str | None = typer.Option(
        None,
        "--slurm-account",
        help="Slurm account name",
    ),
    no_container_mount_home: bool = typer.Option(
        False,
        "--no-container-mount-home",
        help="Disable mounting user home inside container (gives full /dev/shm)",
    ),
    container_env: list[str] | None = typer.Option(
        None,
        "--container-env",
        help="Set container environment variable (repeatable, e.g. --container-env NCCL_TIMEOUT=3600)",
    ),
    verbosity: int = typer.Option(
        2,
        "-v",
        "--verbosity",
        help="Verbosity level (0-4)",
    ),
):
    """
    Submit a bash script as a compute job.

    This command reads a bash script file and submits it to the configured
    compute backend. Returns the job ID if successful.

    Example (Ray):
        llmzero submit-script train.sh -r my-ray-cluster-head-svc

    Example (conda):
        llmzero submit-script train.sh --compute-backend conda --conda-env my_rl_env

    Example (docker):
        llmzero submit-script train.sh --compute-backend docker --docker-image my-training:latest --mount-dir /data --mount-dir /models

    Example (slurmdocker):
        llmzero submit-script train.sh --compute-backend slurmdocker --slurm-partition gpu --docker-image my-training.sqsh
    """
    from autogluon.llmzero.execution.executor import CodeExecutor
    from autogluon.llmzero.rich_logging import configure_logging

    configure_logging(verbosity)

    if not script_path.exists():
        typer.echo(f"Error: Script file not found: {script_path}", err=True)
        raise typer.Exit(code=1)

    compute_backend_config = _build_compute_backend_config(
        compute_backend,
        ray_namespace=ray_namespace,
        ray_service_name=ray_service_name,
        ray_dashboard_port=ray_dashboard_port,
        conda_env=conda_env,
        num_gpus=num_gpus,
        docker_image=docker_image,
        mount_dirs=mount_dirs,
        slurm_partition=slurm_partition,
        slurm_num_nodes=slurm_num_nodes,
        slurm_gpus_per_node=slurm_gpus_per_node,
        slurm_time_limit=slurm_time_limit,
        slurm_account=slurm_account,
        no_container_mount_home=no_container_mount_home,
        container_env=container_env,
    )

    executor = CodeExecutor(
        language="bash",
        timeout=360000,  # 100 hours timeout for job submission
        compute_backend_config=compute_backend_config,
    )

    typer.echo(f"Submitting script: {script_path}")
    success, job_id, message = executor.submit_script(str(script_path))

    if success:
        typer.echo(f"✓ {message}")
        if job_id:
            typer.echo(f"Job ID: {job_id}")
            print(job_id)
        raise typer.Exit(code=0)
    else:
        typer.echo(f"✗ {message}", err=True)
        raise typer.Exit(code=1)


@app.command(name="stop-job")
def stop_job(
    job_id: str = typer.Argument(..., help="Job ID to stop (raysubmit_*, PID, container ID, or Slurm job ID)"),
    compute_backend: str = typer.Option(
        "raycluster",
        "--compute-backend",
        help="Compute backend: 'raycluster', 'conda', 'docker', 'slurm', or 'slurmdocker'",
    ),
    # --- Ray Cluster options ---
    ray_namespace: str = typer.Option(
        "llmzero",
        "--ray-namespace",
        help="Kubernetes namespace for Ray cluster",
    ),
    ray_service_name: str | None = typer.Option(
        None,
        "-r",
        "--ray-service",
        help="Ray cluster head service name",
    ),
    ray_dashboard_port: int = typer.Option(
        8265,
        "--ray-port",
        help="Ray dashboard port",
    ),
    # --- Conda backend options ---
    conda_env: str | None = typer.Option(
        None,
        "--conda-env",
        help="Conda environment name (required for conda backend stop)",
    ),
    # --- Docker backend options ---
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Docker image (required for docker backend stop)",
    ),
    mount_dirs: list[str] | None = typer.Option(
        None,
        "--mount-dir",
        help="Host directory to identity-mount into docker container (repeatable)",
    ),
    # --- Slurm+Docker backend options (slurmdocker) ---
    slurm_partition: str | None = typer.Option(
        None,
        "--slurm-partition",
        help="Slurm partition name (required for slurmdocker backend stop)",
    ),
    verbosity: int = typer.Option(
        2,
        "-v",
        "--verbosity",
        help="Verbosity level (0-4)",
    ),
):
    """
    Stop a running job by its ID.

    For Ray backend, the job ID is a raysubmit_* string.
    For conda backend, the job ID is a process PID.
    For docker backend, the job ID is a container ID.
    For slurmdocker backend, the job ID is a numeric Slurm job ID.

    Example (Ray):
        llmzero stop-job raysubmit_abc123 -r my-ray-cluster-head-svc

    Example (conda):
        llmzero stop-job 12345 --compute-backend conda --conda-env my_rl_env

    Example (docker):
        llmzero stop-job abc123def --compute-backend docker --docker-image my-training:latest

    Example (slurmdocker):
        llmzero stop-job 67890 --compute-backend slurmdocker --slurm-partition gpu
    """
    from autogluon.llmzero.compute_infra import create_backend
    from autogluon.llmzero.rich_logging import configure_logging

    configure_logging(verbosity)

    # Ray-specific format warning
    if compute_backend == "raycluster" and not job_id.startswith("raysubmit_"):
        typer.echo(
            f"Warning: Job ID '{job_id}' doesn't match expected Ray format 'raysubmit_*'",
            err=True,
        )

    compute_backend_config = _build_compute_backend_config(
        compute_backend,
        ray_namespace=ray_namespace,
        ray_service_name=ray_service_name,
        ray_dashboard_port=ray_dashboard_port,
        conda_env=conda_env,
        docker_image=docker_image,
        mount_dirs=mount_dirs,
        slurm_partition=slurm_partition,
    )

    backend = create_backend(compute_backend_config)

    typer.echo(f"Stopping job: {job_id}")
    if backend.terminate(job_id=job_id):
        typer.echo(f"✓ Stopped job: {job_id}")
    else:
        typer.echo(f"✗ Failed to stop job: {job_id}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
