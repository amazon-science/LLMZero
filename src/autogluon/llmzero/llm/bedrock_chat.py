import logging
import os
from typing import Any, Dict, List

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock

from .base_chat import BaseAssistantChat

logger = logging.getLogger(__name__)


class AssistantChatBedrock(ChatBedrock, BaseAssistantChat):
    """Bedrock chat model with LangGraph support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_conversation(self)

    def describe(self) -> Dict[str, Any]:
        base_desc = super().describe()
        return {**base_desc, "model": self.model_id}


def get_bedrock_models() -> List[str]:
    if "AWS_DEFAULT_REGION" not in os.environ:
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
        logger.info("AWS_DEFAULT_REGION not found is os.environ. Set to default value us-west-2.")

    try:
        bedrock = boto3.client("bedrock")
        response = bedrock.list_foundation_models()
        return [model["modelId"] for model in response["modelSummaries"]]
    except Exception as e:
        logger.error(f"Error fetching Bedrock models: {e}")
        return []


def _build_thinking_kwargs(config) -> Dict[str, Any]:
    """Build the extended-thinking model_kwargs fields for Anthropic-on-Bedrock.

    Returns the fields to merge into ``model_kwargs`` (empty dict if thinking is
    disabled). Targets adaptive-thinking models (Claude Opus 4.8, Claude 5 family):
    ``thinking={"type": "adaptive"}`` plus ``output_config={"effort": ...}`` where
    effort is one of ``low|medium|high|xhigh|max``. Adaptive thinking self-caps the
    trace length, so higher effort is safe.

    Thinking is enabled by default for Bedrock. Set ``thinking.enabled: false``
    (or ``thinking: false``) in the LLM config to turn it off.
    """
    thinking_cfg = getattr(config, "thinking", None)

    # Allow `thinking: false` / `thinking: true` as a shorthand for enabled flag.
    if isinstance(thinking_cfg, bool):
        enabled = thinking_cfg
        thinking_cfg = None
    elif thinking_cfg is None:
        enabled = True  # default-on for Bedrock
    else:
        enabled = bool(getattr(thinking_cfg, "enabled", True))

    if not enabled:
        return {}

    fields: Dict[str, Any] = {"thinking": {"type": "adaptive"}}
    effort = getattr(thinking_cfg, "effort", "xhigh") if thinking_cfg is not None else "xhigh"
    if effort is not None:
        fields["output_config"] = {"effort": str(effort)}
    return fields


def create_bedrock_chat(config, session_name: str) -> AssistantChatBedrock:
    """Create a Bedrock chat model instance."""
    model = config.model

    logger.info(f"Using Bedrock model: {model} for session: {session_name}")
    if "AWS_DEFAULT_REGION" not in os.environ:
        raise ValueError("AWS_DEFAULT_REGION key not found in environment")

    # Configure read timeout to 300 seconds
    boto_config = Config(read_timeout=300)

    # Some newer Bedrock models (e.g. Claude Opus 4.8) deprecate `temperature`
    # and reject requests that include it. Treat a null/absent temperature as
    # "omit it" so those models work without per-call special-casing.
    model_kwargs: Dict[str, Any] = {"max_tokens": config.max_tokens}
    temperature = getattr(config, "temperature", None)

    # Extended thinking. When enabled, the API forbids any temperature != 1, so
    # we omit temperature entirely (see thinking_probe: "`temperature` may only
    # be set to 1 when thinking is enabled").
    thinking_kwargs = _build_thinking_kwargs(config)
    if thinking_kwargs:
        model_kwargs.update(thinking_kwargs)
        logger.info(f"Extended thinking enabled for {model}: {thinking_kwargs}")
    elif temperature is not None:
        model_kwargs["temperature"] = temperature

    return AssistantChatBedrock(
        model_id=model,
        model_kwargs=model_kwargs,
        region_name=os.environ["AWS_DEFAULT_REGION"],
        verbose=config.verbose,
        session_name=session_name,
        config=boto_config,
    )
