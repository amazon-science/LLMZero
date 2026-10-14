import logging
import re

logger = logging.getLogger(__name__)


def _strip_thinking_tags(response: str) -> str:
    """
    Remove <think>...</think> tags and their content from the response.

    Args:
        response: The raw response string that may contain thinking tags

    Returns:
        The response with thinking tags and their content removed
    """
    # Remove <think>...</think> tags and their content
    pattern = r"<think>.*?</think>"
    cleaned = re.sub(pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
    return cleaned


def _extract_python_script(response):
    # Look for Python code blocks in the response
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        return None


def _extract_bash_script(response):
    # Look for Bash code blocks in the response
    pattern = r"```bash\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        return None


def extract_code(response, language):
    # First, strip out any thinking tags from the response
    response = _strip_thinking_tags(response)

    result = None

    if language == "python":
        result = _extract_python_script(response)
    elif language == "bash":
        result = _extract_bash_script(response)
    else:
        logger.warning(f"Unsupported language: {language}, trying generic extraction")

    # If language-specific extraction failed, fallback to generic code blocks
    if result is None:
        logger.warning(f"No code block found for {language}, looking for the code wrapped without language specified")
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            result = matches[0].strip()

    # If still nothing found, return the full response
    if result is None:
        logger.warning(f"No code block found, return the full response instead: {response}")
        result = response

    return result
