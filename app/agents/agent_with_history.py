import time
from typing import Dict, List
from app.config import get_api_key
from app.exceptions import LlmClientError
from app.handlers.llm import BaseLlmHandler, get_llm_client
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


def __read_usage(
    llm_response: dict, sent_messages: List[Dict[str, str]], reply: str
) -> TokenUsage:
    """
    Read token usage from the response, or estimate it if it is missing.

    Parameters
    ----------
    llm_response : dict
        The JSON response from the LLM.
    sent_messages : List[Dict[str, str]]
        The chat messages that were sent, used for the estimate. The
        default system prompt is added the same way as for the request.
    reply : str
        The reply text, used for the estimate.

    Returns
    -------
    TokenUsage
        Usage reported by the provider, or an estimate marked as estimated.
    """
    usage = llm_response.get("usage") or {}
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


async def call_agent_with_history(
    provider: ProviderConfig,
    user_query: str,
    use_history: bool = True,
    **additional_kwargs,
) -> LlmAnswer:
    """
    Send the user query to the LLM along with the session chat history.

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
    global SESSION_CONV_HIST
    client = get_llm_client(
        base_url=provider.base_url,
        api_key=get_api_key(provider),
        model=provider.model,
        context_window=provider.context_window,
        reserved_output_tokens=provider.reserved_output_tokens,
    )

    # Add the user message and maintain context windows. The session history
    # is only updated once the LLM replies, so a failed call can be retried.
    previous_turns = SESSION_CONV_HIST if use_history else []
    chat_history = __maintain_chat_window(
        [*previous_turns, {"role": "user", "content": user_query}],
        max_tokens=__history_budget(provider),
    )

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
    if use_history:
        SESSION_CONV_HIST = [
            *chat_history,
            {"role": "assistant", "content": assistant_reply},
        ]

    usage = __read_usage(llm_async_response, chat_history, assistant_reply)
    return LlmAnswer(
        text=assistant_reply,
        provider=provider.name,
        model=provider.model,
        usage=usage,
        latency_seconds=latency_seconds,
        cost=calculate_cost(usage, provider.prices),
    )
