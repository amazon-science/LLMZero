import logging

from ..execution import CodeExecutor
from ..prompts import ExecuterPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class ExecuterAgent(BaseAgent):
    """
    Execute the code and give analysis.

    Agent Input:
    - Code to execute
    - Language
    - Execution task description
    - Execution data

    Agent Output:
    - Execution decision
    - Error summary
    - Validation score
    - Execution logs
    """

    def __init__(
        self,
        config,
        manager,
        language,
        timeout,
        executer_llm_config,
        executer_prompt_template,
        validation_score_mode=None,
    ):
        super().__init__(config=config, manager=manager)
        assert language in ["bash", "python"]

        self.timeout = timeout
        self.language = language
        self.executer_llm_config = executer_llm_config

        self.output_folder = manager.output_folder if hasattr(manager, "output_folder") else None

        # Validation score mode: "best_validation" or "final_validation".
        # Resolved from config (the single source of truth); the explicit param,
        # when given, overrides config (used by tests).
        self.validation_score_mode = validation_score_mode or getattr(
            self.config, "validation_score_mode", "best_validation"
        )

        if executer_prompt_template is not None:
            self.executer_prompt_template = executer_prompt_template
        elif self.executer_llm_config.template is not None:
            self.executer_prompt_template = self.executer_llm_config.template
        else:
            self.executer_prompt_template = None

        if self.executer_llm_config.multi_turn:
            self.executer_llm = init_llm(
                llm_config=self.executer_llm_config,
                agent_name=f"{language}_executer",
                multi_turn=self.executer_llm_config.multi_turn,
            )

        self.executer_prompt = ExecuterPrompt(
            llm_config=self.executer_llm_config, manager=manager, template=self.executer_prompt_template
        )

    def __call__(self, code_to_execute, code_to_analyze=None, execution_task=None, execution_data=None):

        self.manager.log_agent_start("ExecuterAgent: executing code and collecting stdout/stderr for evaluation.")

        if code_to_analyze is None:
            code_to_analyze = code_to_execute

        # Create executor with compute backend config from manager
        compute_backend_config = dict(self.manager.config.compute_infra)

        executor = CodeExecutor(
            language=self.language, timeout=self.timeout, compute_backend_config=compute_backend_config
        )

        # Build the observer set via the configured training framework. The
        # framework owns observer wiring (parser, grammar) so core execution
        # code stays framework-agnostic.
        from ..training_frameworks import get_framework_from_config
        from ..training_frameworks.observer_context import ObserverContext

        framework = get_framework_from_config(self.config)

        validation_metric_pattern = getattr(self.config, "validation_metric_pattern", None)

        # Both validation modes need a metric pattern (config key or framework default).
        # Guard early here with an actionable message rather than deep inside build_observers.
        if self.validation_score_mode in ("best_validation", "final_validation"):
            resolved_pattern = validation_metric_pattern or framework.validation_metric_pattern()
            if not resolved_pattern:
                raise ValueError(
                    f"validation_metric_pattern is required when using {self.validation_score_mode} mode. "
                    "Please set validation_metric_pattern in your config YAML file. "
                    "Example: validation_metric_pattern: 'val-core/.*__dev'"
                )
            logger.info(f"Using validation metric pattern: {resolved_pattern}")

        early_stopper_config = getattr(self.config, "early_stopper", {})
        ckpt_config = getattr(self.config, "checkpoint_optimization", None)
        is_from_scratch = getattr(self.manager.current_node, "from_scratch", True)
        if is_from_scratch:
            logger.info("Skipping early stopping for train-from-scratch node")

        early_stopper_callback = None
        early_stopper_enabled = getattr(early_stopper_config, "enabled", False)
        if (
            early_stopper_enabled
            and not is_from_scratch
            and hasattr(self.manager, "early_stoppers")
            and self.manager.early_stoppers
        ):

            def early_stopper_callback(current_metrics, best_metrics, best_score, elapsed_time):
                return self.manager.get_early_stop_decision(current_metrics, elapsed_time)

        ctx = ObserverContext(
            validation_score_mode=self.validation_score_mode,
            validation_metric_pattern=validation_metric_pattern,
            checkpoint_optimization_enabled=bool(ckpt_config and getattr(ckpt_config, "enabled", False)),
            early_stopper_enabled=early_stopper_enabled,
            early_stopper_callback=early_stopper_callback,
            best_run_metrics=getattr(self.manager, "_best_run_metrics_history", None),
            best_run_validation_score=getattr(self.manager, "_best_validation_score", None),
            min_steps_before_early_stop=getattr(early_stopper_config, "min_steps_before_early_stop", 0),
            checkpoint_interval_seconds=getattr(early_stopper_config, "checkpoint_interval_seconds", 1800),
            is_from_scratch=is_from_scratch,
        )

        observers = framework.build_observers(ctx)
        for observer in observers:
            executor.attach_observer(observer)
        logger.info(f"Attached {len(observers)} observers via {type(framework).__name__}")

        # Execute code
        logger.brief(f"Executing {self.language} code with {self.validation_score_mode} validation mode")
        success, stdout, stderr, observer_results = executor.execute(code_to_execute)

        # Process observer results to extract validation score
        validation_score = None

        # Always terminate jobs after execution completes (timeout, early stop, or natural completion)
        logger.info("Terminating jobs after execution")
        executor.terminate_last_job()
        # Clear backend cache after execution completes
        # logger.info("Cleaning up backend cache")
        # executor.cleanup_backend()

        # Extract the node's validation score from the validation observer. Both
        # best_validation and final_validation modes report it under "validation"
        # (the observer's reduction decides best-vs-last).
        val_result = observer_results.get("validation", {})
        validation_score = val_result.get("validation_score")
        if validation_score is not None:
            logger.info(f"Using {self.validation_score_mode} score: {validation_score}")
        else:
            logger.warning("Could not extract validation score from output.")

        # Analyze execution results with LLM
        logger.info("Analyzing execution results with LLM")
        prompt = self.executer_prompt.build(
            stdout=stdout,
            stderr=stderr,
            code_to_analyze=code_to_analyze,
            execution_task=execution_task,
            execution_data=execution_data,
        )

        # Initialize LLM if not multi_turn
        if not self.executer_llm_config.multi_turn:
            self.executer_llm = init_llm(
                llm_config=self.executer_llm_config,
                agent_name=f"{self.language}_executer",
                multi_turn=self.executer_llm_config.multi_turn,
            )

        # Get LLM analysis
        response = self.executer_llm.assistant_chat(prompt)

        # Parse the LLM response
        decision, error_summary, llm_validation_score = self.executer_prompt.parse(response)

        # Use observer validation score if available, otherwise use LLM-extracted score
        resolved_validation_score = validation_score if validation_score is not None else llm_validation_score

        return decision, error_summary, resolved_validation_score, prompt, stderr, stdout, observer_results
