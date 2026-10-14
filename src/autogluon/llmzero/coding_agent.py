import logging
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

from .constants import DEFAULT_CONFIG_PATH, DEFAULT_RUNS_DIR, generate_run_folder_name
from .rich_logging import configure_logging

logger = logging.getLogger(__name__)


def run_agent(
    input_data_folder,
    output_folder=None,
    config_path=None,
    max_iterations=16,
    stop_after_success=None,
    enable_per_iteration_instruction=False,
    initial_user_input=None,
    compute_infra_overrides=None,
    wandb_api_key=None,
    wandb_project_postfix="",
    run_postfix="",
    job_parallel=1,
    validation_score_mode=None,
    manager=None,
    verbosity=1,
):
    """
    Run the AutoGluon Assistant with MCTS-based search strategy.

    Args:
        input_data_folder: Path to input data directory
        output_folder: Path to output directory
        config_path: Path to configuration file
        max_iterations: Maximum number of iterations
        stop_after_success: Whether to stop after finding the first valid solution
        enable_per_iteration_instruction: Whether to ask for user input at each iteration
        initial_user_input: Initial user instruction
        compute_infra_overrides: Optional dict with compute infra config overrides
        wandb_api_key: Optional WandB API key for experiment tracking
        wandb_project_postfix: Optional postfix for wandb project names
        run_postfix: Optional postfix for output folder names
        job_parallel: Number of parallel jobs; >1 enables async search mode
        validation_score_mode: "best_validation" or "final_validation"; overrides config when given
        verbosity: Verbosity level

    Returns:
        None
    """
    if output_folder is None or not output_folder:
        output_folder = os.path.join(str(DEFAULT_RUNS_DIR), generate_run_folder_name())

    # Create output directory
    output_dir = Path(output_folder).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=True)

    configure_logging(verbosity=verbosity, output_dir=output_dir)
    from .managers.node_manager import NodeManager

    # Always load default config first
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Default config file not found: {DEFAULT_CONFIG_PATH}")

    config = OmegaConf.load(DEFAULT_CONFIG_PATH)

    # If config_path is provided, merge it with the default config
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)

    if stop_after_success is not None:
        config.stop_after_success = stop_after_success

    if validation_score_mode is not None:
        config.validation_score_mode = validation_score_mode

    # --- Compute infrastructure config ---
    if compute_infra_overrides:
        if not hasattr(config, "compute_infra"):
            config.compute_infra = OmegaConf.create({"type": "raycluster"})

        for key, value in compute_infra_overrides.items():
            config.compute_infra[key] = value

        logger.info(f"Applied compute infra overrides: {compute_infra_overrides}")

    # Validate via the registered backend (delegates infra-specific checks)
    if hasattr(config, "compute_infra"):
        from .compute_infra import validate_backend_config

        validate_backend_config(dict(config.compute_infra))

    # Validate the training framework selection (independent of compute infra)
    from .training_frameworks import validate_framework_config

    validate_framework_config(config)

    if manager is None:
        if job_parallel > 1:
            from .managers.async_node_manager import AsyncNodeManager

            manager = AsyncNodeManager(
                input_data_folder=input_data_folder,
                output_folder=output_folder,
                config=config,
                initial_user_input=initial_user_input,
                enable_per_iteration_instruction=enable_per_iteration_instruction,
                max_concurrent=job_parallel,
                wandb_api_key=wandb_api_key,
                wandb_project_postfix=wandb_project_postfix,
                run_postfix=run_postfix,
            )
        else:
            manager = NodeManager(
                input_data_folder=input_data_folder,
                output_folder=output_folder,
                config=config,
                enable_per_iteration_instruction=enable_per_iteration_instruction,
                initial_user_input=initial_user_input,
                wandb_api_key=wandb_api_key,
                wandb_project_postfix=wandb_project_postfix,
                run_postfix=run_postfix,
            )

    # Initialize the manager (generate initial prompts)
    manager.initialize()

    start_time = time.time()

    if job_parallel > 1:
        # Async search: concurrent jobs with Top-K Weighted Deepening
        manager.run_search(max_iterations=max_iterations)
        manager.create_best_run_copy()
    else:
        # Sequential MCTS search
        iteration = 0
        while iteration < max_iterations:
            logger.brief(f"Starting MCTS iteration {iteration + 1}/{max_iterations}")

            success = manager.step()

            if success:
                manager.create_best_run_copy()
                if config.stop_after_success:
                    logger.brief("Stopping search - solution found and stop_after_success is enabled")
                    break
            elif success is None:
                logger.brief("Stopping search - all nodes are terminal.")
                break

            iteration += 1

            if iteration >= max_iterations:
                logger.warning(f"[bold red]Warning: Reached maximum iterations ({max_iterations})[/bold red]")

    manager.visualize_results()
    manager.report_token_usage()
    manager.cleanup_all_checkpoints()
    # Report token usage and validation score summary
    manager.cleanup()

    # Log summary
    elapsed_time = time.time() - start_time
    logger.brief(f"MCTS search completed in {elapsed_time:.2f} seconds")
    logger.brief(f"Total nodes explored: {manager.time_step + 1}")
    logger.brief(f"Best validation score: {manager.best_validation_score}")
    logger.brief(f"Tools used: {', '.join(getattr(manager, 'used_tools', []))}")
    logger.brief(f"Output saved in {output_dir}")
