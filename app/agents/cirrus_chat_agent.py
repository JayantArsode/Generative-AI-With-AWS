import time
from typing import AsyncGenerator, Dict, List
from app.config import get_api_key
from app.exceptions import LlmClientError
from app.handlers.llm import BaseLlmHandler, get_llm_client
from app.handlers.llm.llm_client_handlers import HTTPLlmClientHandler
from app.prompts.prompt_loaders import SystemPrompts, get_system_prompt
from app.schemas.llm_response import LlmAnswer, TokenUsage
from app.schemas.provider_config import ProviderConfig
from app.utils.cost import calculate_cost
from app.utils.tokens import (
    count_message_tokens,
    count_single_message_tokens,
    count_text_tokens,
)

SESSION_CONV_HIST = []


def __maintain_chat_window(
    chat_input_with_history: List[Dict[str, str]], max_tokens: int = 200
) -> List[Dict[str, str]]:
    """
    Trim a chat history list to stay under a specific token limit.

    Messages are counted from newest to oldest, and older messages are dropped
    once the budget is reached. The newest message is always kept, even if it
    alone exceeds the budget, so the client can refuse it with a clear message.

    Parameters
    ----------
    chat_input_with_history : List[Dict[str, str]]
        Chat history in chronological order,
        e.g. ``[{"role": "user", "content": "Hello"}]``.
    max_tokens : int, optional
        The maximum allowable tokens for the messages, by default 200.

    Returns
    -------
    List[Dict[str, str]]
        The newest messages that fit in the budget, in chronological order.
    """
    total_tokens = 0
    chat_input_with_history_trimmed = []

    # Process from newest to oldest to preserve the most recent context
    for msg in reversed(chat_input_with_history):
        msg_tokens = count_single_message_tokens(msg)

        # Check if adding this message exceeds our safety budget,
        # but always keep the newest message
        if chat_input_with_history_trimmed and total_tokens + msg_tokens > max_tokens:
            break  # Stop adding older messages

        total_tokens += msg_tokens
        chat_input_with_history_trimmed.append(msg)

    # Reverse back to keep the original chronological order
    chat_input_with_history_trimmed.reverse()

    return chat_input_with_history_trimmed


def __history_budget(provider: ProviderConfig) -> int:
    """
    Tokens left for the chat history once the system prompt is counted.

    Parameters
    ----------
    provider : ProviderConfig
        The provider whose context window to use.

    Returns
    -------
    int
        Input tokens the history may use.
    """
    system_message = {
        "role": "system",
        "content": get_system_prompt(SystemPrompts.DEFAULT_PROMPT),
    }
    return provider.max_input_tokens - count_message_tokens([system_message])


def __build_client(provider: ProviderConfig) -> HTTPLlmClientHandler:
    """
    Get the LLM client for a provider.

    Parameters
    ----------
    provider : ProviderConfig
        The provider and model to call.

    Returns
    -------
    HTTPLlmClientHandler
        A cached client with the provider's URL, key and context window.

    Raises
    ------
    ProviderConfigError
        If the provider's API key is not set.
    """
    return get_llm_client(
        base_url=provider.base_url,
        api_key=get_api_key(provider),
        model=provider.model,
        context_window=provider.context_window,
        reserved_output_tokens=provider.reserved_output_tokens,
    )


def __build_chat_history(
    provider: ProviderConfig, user_query: str, use_history: bool
) -> List[Dict[str, str]]:
    """
    Build the messages to send: earlier turns (if any) plus the new query.

    Parameters
    ----------
    provider : ProviderConfig
        The provider whose context window limits the history.
    user_query : str
        The new message from the user.
    use_history : bool
        Include the earlier turns from ``SESSION_CONV_HIST``.

    Returns
    -------
    List[Dict[str, str]]
        The trimmed messages, newest last.
    """
    previous_turns = SESSION_CONV_HIST if use_history else []
    return __maintain_chat_window(
        [*previous_turns, {"role": "user", "content": user_query}],
        max_tokens=__history_budget(provider),
    )


def __save_turn(
    chat_history: List[Dict[str, str]], assistant_reply: str, use_history: bool
) -> None:
    """
    Save the sent messages and the reply as the new session history.

    Called only after a successful reply, so a failed call can be retried.

    Parameters
    ----------
    chat_history : List[Dict[str, str]]
        The messages that were sent.
    assistant_reply : str
        The full reply from the LLM.
    use_history : bool
        Nothing is saved when history is off.
    """
    global SESSION_CONV_HIST
    if use_history:
        SESSION_CONV_HIST = [
            *chat_history,
            {"role": "assistant", "content": assistant_reply},
        ]


def __read_usage(
    usage: dict | None, sent_messages: List[Dict[str, str]], reply: str
) -> TokenUsage:
    """
    Read token usage reported by the provider, or estimate it if missing.

    Parameters
    ----------
    usage : dict | None
        The ``usage`` object from the response or the last streamed chunk,
        e.g. ``{"prompt_tokens": 22, "completion_tokens": 87}``.
    sent_messages : List[Dict[str, str]]
        The chat messages that were sent, used for the estimate. The
        default system prompt is added the same way as for the request.
    reply : str
        The full reply text, used for the estimate.

    Returns
    -------
    TokenUsage
        Usage reported by the provider, or an estimate marked as estimated.
    """
    usage = usage or {}
    if isinstance(usage.get("prompt_tokens"), int):
        details = usage.get("prompt_tokens_details") or {}
        return TokenUsage(
            input_tokens=usage["prompt_tokens"],
            cached_input_tokens=details.get("cached_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
        )

    return TokenUsage(
        input_tokens=count_message_tokens(BaseLlmHandler.to_message(sent_messages)),
        output_tokens=count_text_tokens(reply),
        estimated=True,
    )


def __read_stream_chunk(chunk: dict) -> tuple[str, dict | None]:
    """
    Take the answer text and the usage out of one streamed chunk.

    Parameters
    ----------
    chunk : dict
        One chunk from ``astream``, e.g.
        ``{"choices": [{"delta": {"content": "Hel"}}], "usage": null}``.

    Returns
    -------
    tuple[str, dict | None]
        The text in this chunk (empty if none, e.g. a reasoning-only or
        usage-only chunk) and the usage object if the chunk has one.

    Raises
    ------
    LlmClientError
        If the provider streamed an error, or the chunk has an unexpected shape.
    """
    if "error" in chunk:
        raise LlmClientError(f"LLM stream returned an error: {chunk['error']}")

    try:
        choices = chunk["choices"]
        # The usage-only chunk at the end has an empty choices list
        delta = (choices[0].get("delta") or {}) if choices else {}
        text = delta.get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise LlmClientError(f"Unexpected LLM stream chunk: {chunk}") from e

    usage = chunk.get("usage")
    return text, usage if isinstance(usage, dict) else None


def __tokens_per_second(
    reply: str, first_token_at: float | None, finished_at: float
) -> float | None:
    """
    Speed of the answer after its first token.

    Parameters
    ----------
    reply : str
        The full reply text.
    first_token_at : float | None
        ``time.perf_counter()`` when the first text arrived.
    finished_at : float
        ``time.perf_counter()`` when the stream ended.

    Returns
    -------
    float | None
        Tokens of answer text per second after the first token, or ``None``
        if there was no text or no time passed to measure.
    """
    if first_token_at is None:
        return None

    generation_seconds = finished_at - first_token_at
    tokens_after_first = count_text_tokens(reply) - 1
    if generation_seconds <= 0 or tokens_after_first <= 0:
        return None
    return tokens_after_first / generation_seconds


async def call_agent(
    provider: ProviderConfig,
    user_query: str,
    use_history: bool = True,
    **additional_kwargs,
) -> LlmAnswer:
    """
    Send the user query to the LLM and wait for the full reply.

    With history on, the user query and the LLM reply are both saved in
    ``SESSION_CONV_HIST``, and the history is trimmed to fit the context
    window before each call. With history off, only the new query is sent
    and nothing is saved, so the model does not know what was asked before.

    Parameters
    ----------
    provider : ProviderConfig
        The provider and model to call.
    user_query : str
        The new message from the user.
    use_history : bool, optional
        Send and save the chat history, by default True.
    **additional_kwargs
        Extra fields added to the request payload, e.g. ``temperature``.

    Returns
    -------
    LlmAnswer
        The reply with its token usage, latency and cost.

    Raises
    ------
    ProviderConfigError
        If the provider's API key is not set.
    ValueError
        If the user query or chat history is not a valid prompt.
    ContextWindowExceededError
        If the query does not fit the context window. Nothing is sent.
    LlmClientError
        If the LLM call fails or the response does not contain a reply.
        The session history is left unchanged in this case.
    """
    client = __build_client(provider)
    chat_history = __build_chat_history(provider, user_query, use_history)

    # Call the llm and time it
    started_at = time.perf_counter()
    llm_async_response = await client.ainvoke(chat_history, **additional_kwargs)
    latency_seconds = time.perf_counter() - started_at

    try:
        assistant_reply = llm_async_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LlmClientError(
            f"Unexpected LLM response format: {llm_async_response}"
        ) from e

    # Add the assistant reply so the next call has the full conversation
    __save_turn(chat_history, assistant_reply, use_history)

    usage = __read_usage(llm_async_response.get("usage"), chat_history, assistant_reply)
    return LlmAnswer(
        text=assistant_reply,
        provider=provider.name,
        model=provider.model,
        usage=usage,
        latency_seconds=latency_seconds,
        cost=calculate_cost(usage, provider.prices),
    )


async def call_agent_with_stream(
    provider: ProviderConfig,
    user_query: str,
    use_history: bool = True,
    **additional_kwargs,
) -> AsyncGenerator[str | LlmAnswer]:
    """
    Send the user query to the LLM and yield the reply as it is generated.

    History works the same as in ``call_agent``. The turn is saved only when
    the whole stream arrived, so a stream that fails or is stopped halfway
    leaves the history unchanged.

    Parameters
    ----------
    provider : ProviderConfig
        The provider and model to call.
    user_query : str
        The new message from the user.
    use_history : bool, optional
        Send and save the chat history, by default True.
    **additional_kwargs
        Extra fields added to the request payload, e.g. ``temperature``.

    Yields
    ------
    str | LlmAnswer
        Pieces of answer text as they arrive, then one final ``LlmAnswer``
        with the full text, usage, latency, cost, time to first token and
        tokens per second.

    Raises
    ------
    ProviderConfigError
        If the provider's API key is not set.
    ValueError
        If the user query or chat history is not a valid prompt.
    ContextWindowExceededError
        If the query does not fit the context window. Nothing is sent.
    LlmClientError
        If the LLM call fails or streams an error or an unexpected chunk.
        The session history is left unchanged in this case.
    """
    client = __build_client(provider)
    chat_history = __build_chat_history(provider, user_query, use_history)

    reply_parts = []
    usage = None
    first_token_at = None

    # Call the llm and time it
    started_at = time.perf_counter()
    async for chunk in client.astream(chat_history, **additional_kwargs):
        text, chunk_usage = __read_stream_chunk(chunk)
        if chunk_usage is not None:
            usage = chunk_usage
        if text:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            reply_parts.append(text)
            yield text
    finished_at = time.perf_counter()

    assistant_reply = "".join(reply_parts)

    # Add the assistant reply so the next call has the full conversation
    __save_turn(chat_history, assistant_reply, use_history)

    token_usage = __read_usage(usage, chat_history, assistant_reply)
    yield LlmAnswer(
        text=assistant_reply,
        provider=provider.name,
        model=provider.model,
        usage=token_usage,
        latency_seconds=finished_at - started_at,
        cost=calculate_cost(token_usage, provider.prices),
        time_to_first_token_seconds=(
            first_token_at - started_at if first_token_at is not None else None
        ),
        tokens_per_second=__tokens_per_second(
            assistant_reply, first_token_at, finished_at
        ),
    )
