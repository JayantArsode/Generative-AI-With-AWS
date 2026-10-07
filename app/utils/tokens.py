from functools import lru_cache
from typing import Dict, List
import tiktoken

ENCODING_NAME = "o200k_base"

# Standard framing overhead per message (varies slightly by model version)
TOKENS_PER_MESSAGE = 3

# Tokens the model uses to start its reply
TOKENS_PER_REPLY = 3


@lru_cache
def get_encoding() -> tiktoken.Encoding:
    """
    Load the tokenizer once and reuse it.

    Returns
    -------
    tiktoken.Encoding
        The ``o200k_base`` tokenizer.
    """
    return tiktoken.get_encoding(ENCODING_NAME)


def count_text_tokens(text: str) -> int:
    """
    Count the tokens in a piece of text.

    This is an estimate for models that are not from OpenAI, because each
    model family has its own tokenizer. It is close enough to check budgets.

    Parameters
    ----------
    text : str
        The text to count.

    Returns
    -------
    int
        Number of tokens.
    """
    return len(get_encoding().encode(text, disallowed_special=()))


def count_single_message_tokens(message: Dict[str, str]) -> int:
    """
    Estimate the tokens of one chat message.

    Parameters
    ----------
    message : Dict[str, str]
        A message like ``{"role": "user", "content": "Hello"}``.

    Returns
    -------
    int
        Tokens for the role and content, plus the framing overhead.
    """
    return (
        TOKENS_PER_MESSAGE
        + count_text_tokens(message["role"])
        + count_text_tokens(message["content"])
    )


def count_message_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Estimate the input tokens of a list of chat messages.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        Messages like ``[{"role": "user", "content": "Hello"}]``.

    Returns
    -------
    int
        Tokens for every message, plus the overhead of starting the reply.
    """
    return TOKENS_PER_REPLY + sum(count_single_message_tokens(m) for m in messages)
