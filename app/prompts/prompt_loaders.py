import yaml
from functools import lru_cache
from enum import Enum
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
SYSTEM_PROMPTS_FILE = PROMPTS_DIR / "system" / "system_prompts.yml"


class SystemPrompts(Enum):
    """
    Keys of the system prompts stored in the system prompts yml file.

    Attributes
    ----------
    DEFAULT_PROMPT : str
        Key of the default system prompt used when none is provided.
    """

    DEFAULT_PROMPT = "DEFAULT_SYSTEM_PROMPT"


@lru_cache
def __load_yml_as_dict(file_path: str) -> dict:
    """
    This will load the any yml file from it's path and convert in python dict.

    Parameters
    ----------
    file_path: str
        This the file path for yml file.

    Returns
    -------
    data: dict
        The yml file converted to dict result.

    Raises
    ------
    FileNotFoundError:
        If file does not exist.
    ValueError:
        In case of any invalid yml file or invalid data in file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The prompt file with name: {file_path} does not exist"
        )
    except yaml.YAMLError as ye:
        raise ValueError(f"Error parsing YAML file: {ye}")

    if not isinstance(prompt_dict, dict):
        raise ValueError(f"Prompt file {file_path} must contain 'key: value' pairs")
    return prompt_dict


def get_system_prompt(prompt_key: SystemPrompts):
    """
    This will return any system prompt added in yml file.

    Parameters
    ----------
    prompt_key: SystemPrompts
        This the enum key for prompt to access example SystemPrompts.DEFAULT_PROMPT

    Returns
    -------
    prompt:
        The prompt with respect to the prompt key provided.
    """
    prompts = __load_yml_as_dict(SYSTEM_PROMPTS_FILE)
    prompt = prompts.get(prompt_key.value)

    if not isinstance(prompt, str) or not prompt.strip():
        raise KeyError(
            f"System prompt '{prompt_key.value}' is missing or empty in {SYSTEM_PROMPTS_FILE}"
        )
    return prompt.strip()
