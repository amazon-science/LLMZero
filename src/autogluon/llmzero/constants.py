import uuid
from datetime import datetime
from pathlib import Path

### PATHs
PACKAGE_ROOT = Path(__file__).parent  # /src/autogluon/assistant
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"
DEFAULT_RUNS_DIR = "runs"


def generate_run_folder_name() -> str:
    """Generate a run folder name in the format: llmzero-YYMMDDHHMMSS-<4_digit_uuid>."""
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    return f"llmzero-{timestamp}-{short_uuid}"


### Default Library
DEFAULT_LIBRARY = "machine learning"


### CLI
DETAIL_LEVEL = 19
BRIEF_LEVEL = 25
CONSOLE_HANDLER = "console_handler"
VERBOSITY_MAP = {
    "DETAIL": "3",
    "INFO": "2",
    "BRIEF": "1",
}


### Provider defaults
PROVIDER_DEFAULTS = {
    "bedrock": "us.anthropic.claude-opus-4-8",
    "openai": "gpt-4o-2024-08-06",
}
