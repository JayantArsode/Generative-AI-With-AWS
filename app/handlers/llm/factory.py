from functools import lru_cache
from app.handlers.llm.llm_client_handlers import HTTPLlmClientHandler


@lru_cache
def get_llm_client(
    base_url: str,
    api_key: str | None,
    model: str,
    context_window: int | None = None,
    reserved_output_tokens: int = 0,
) -> HTTPLlmClientHandler:
    """
    Return a cached LLM client for the given settings.

    Parameters
    ----------
    base_url : str
        Full URL of the LLM chat endpoint.
    api_key : str | None
        API key used to authenticate with the LLM provider, or ``None`` if
        the provider needs no key.
    model : str
        Name of the LLM model to call.
    context_window : int | None, optional
        Maximum tokens the model accepts. ``None`` skips the size check.
    reserved_output_tokens : int, optional
        Tokens kept free in the context window for the answer, by default 0.

    Returns
    -------
    HTTPLlmClientHandler
        Client that calls the LLM over HTTP. The same instance is returned
        for the same arguments.
    """
    return HTTPLlmClientHandler(
        base_url=base_url,
        api_key=api_key,
        model=model,
        context_window=context_window,
        reserved_output_tokens=reserved_output_tokens,
    )
