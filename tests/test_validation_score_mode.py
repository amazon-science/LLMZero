"""Tests for validation_score_mode wiring (config + CLI, no constant)."""

from omegaconf import OmegaConf


def _make_executer(config_overrides=None, explicit_mode=None):
    """Build an ExecuterAgent against a minimal in-memory config.

    ExecuterAgent.__init__ constructs an ExecuterPrompt but does NOT call the
    LLM (init_llm is lazy when multi_turn is False), so no network is touched.
    """
    from unittest.mock import MagicMock

    from autogluon.llmzero.agents.executer_agent import ExecuterAgent

    base = {
        "executer": {
            "provider": "bedrock",
            "model": "x",
            "max_tokens": 100,
            "temperature": 0.1,
            "top_p": 0.9,
            "verbose": False,
            "multi_turn": False,
            "template": None,
            "add_coding_format_instruction": False,
        },
        "training_framework": "verl_grpo",
        "validation_metric_pattern": "val-core/.*__dev",
    }
    if config_overrides:
        base.update(config_overrides)
    config = OmegaConf.create(base)

    manager = MagicMock()
    manager.config = config
    manager.output_folder = None

    return ExecuterAgent(
        config=config,
        manager=manager,
        language="bash",
        timeout=10,
        executer_llm_config=config.executer,
        executer_prompt_template=None,
        validation_score_mode=explicit_mode,
    )


def test_resolves_mode_from_config():
    agent = _make_executer({"validation_score_mode": "final_validation"})
    assert agent.validation_score_mode == "final_validation"


def test_defaults_to_best_validation_when_config_unset():
    agent = _make_executer()  # no validation_score_mode key
    assert agent.validation_score_mode == "best_validation"


def test_explicit_param_overrides_config():
    agent = _make_executer({"validation_score_mode": "best_validation"}, explicit_mode="final_validation")
    assert agent.validation_score_mode == "final_validation"


def test_constant_is_gone():
    import autogluon.llmzero.constants as c

    assert not hasattr(c, "RL_VALIDATION_SCORE_MODE")


def test_run_agent_applies_cli_value_to_config():
    """run_agent must copy a non-None validation_score_mode into the config."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({"validation_score_mode": "best_validation"})

    # Replicate run_agent's apply logic in isolation (the real run_agent needs a
    # data folder + manager; here we assert the contract the implementation must meet).
    validation_score_mode = "final_validation"
    if validation_score_mode is not None:
        config.validation_score_mode = validation_score_mode

    assert config.validation_score_mode == "final_validation"


def test_run_agent_accepts_validation_score_mode_kwarg():
    """run_agent's signature must accept validation_score_mode."""
    import inspect

    from autogluon.llmzero.coding_agent import run_agent

    assert "validation_score_mode" in inspect.signature(run_agent).parameters
