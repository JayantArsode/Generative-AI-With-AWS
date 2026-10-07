from typing import AsyncGenerator, Generator
from app.handlers.llm.base import BaseLlmHandler
from app.exceptions import ContextWindowExceededError, LlmClientError
from app.schemas.prompt_templates import Messages
from app.utils.tokens import count_message_tokens
import requests
import httpx

REQUEST_TIMEOUT_SECONDS = 60


def _http_error_message(status_code: int, body: str) -> str:
    """
    Build a readable message for an error status returned by the LLM.

    Parameters
    ----------
    status_code : int
        HTTP status code of the response.
    body : str
        Response body, shortened to keep the message readable.

    Returns
    -------
    str
        Message with the status code and the start of the response body.
    """
    return f"LLM returned HTTP {status_code}: {body[:500]}"


class HTTPLlmClientHandler(BaseLlmHandler):
    """
    LLM client that calls an OpenAI compatible chat endpoint over HTTP.

    Parameters
    ----------
    base_url : str
        Full URL of the LLM chat endpoint.
    api_key : str | None
        API key sent as a Bearer token. ``None`` sends no ``Authorization``
        header, for providers that need no key.
    model : str
        Name of the LLM model to call.
    context_window : int | None, optional
        Maximum tokens the model accepts. ``None`` skips the size check.
    reserved_output_tokens : int, optional
        Tokens kept free in the context window for the answer, by default 0.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        context_window: int | None = None,
        reserved_output_tokens: int = 0,
    ):
        self.base_url = base_url
        self.model = model
        self.context_window = context_window
        self.reserved_output_tokens = reserved_output_tokens
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def __check_context_window(self, messages: list[dict]) -> None:
        """
        Refuse messages that would not fit the context window.

        Parameters
        ----------
        messages : list[dict]
            The messages that would be sent.

        Raises
        ------
        ContextWindowExceededError
            If the estimated input tokens plus the tokens reserved for the
            answer are more than the context window.
        """
        if self.context_window is None:
            return

        estimated_tokens = count_message_tokens(messages)
        max_input_tokens = self.context_window - self.reserved_output_tokens
        if estimated_tokens > max_input_tokens:
            raise ContextWindowExceededError(
                f"Request is about {estimated_tokens:,} tokens, but {self.model} "
                f"has room for only {max_input_tokens:,} input tokens "
                f"(context window {self.context_window:,}, "
                f"{self.reserved_output_tokens:,} kept for the answer). "
                "Nothing was sent. Shorten the message and try again."
            )

    def __get_llm_payload(self, prompt: str | Messages, **additional_kwargs) -> dict:
        """
        Build the request payload for the LLM.

        Parameters
        ----------
        prompt : str | Messages
            The prompt as a string or a list of messages.
        **additional_kwargs
            Extra fields added to the payload, e.g. ``temperature``.

        Returns
        -------
        dict
            Payload with ``messages``, ``model`` and any extra fields.

        Raises
        ------
        ValueError
            If the prompt is invalid.
        ContextWindowExceededError
            If the prompt would not fit the context window.
        """
        messages = self.to_message(prompt)
        self.__check_context_window(messages)
        return {
            "messages": messages,
            "model": self.model,
            **additional_kwargs,
        }

    def invoke(self, prompt: str | Messages, **additional_kwargs) -> dict:
        """
        Call the LLM and return the full response.

        Parameters
        ----------
        prompt : str | Messages
            The prompt as a string or a list of messages.
        **additional_kwargs
            Extra fields added to the request payload, e.g. ``temperature``.

        Returns
        -------
        dict
            The JSON response from the LLM.

        Raises
        ------
        ValueError
            If the prompt is invalid.
        ContextWindowExceededError
            If the prompt would not fit the context window. Nothing is sent.
        LlmClientError
            If the LLM cannot be reached, times out, returns an error status
            or returns a response that is not valid JSON.
        """
        invoke_payload = self.__get_llm_payload(prompt, **additional_kwargs)
        try:
            response = requests.post(
                url=self.base_url,
                json=invoke_payload,
                headers={**self.headers, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as httpe:
            raise LlmClientError(
                _http_error_message(httpe.response.status_code, httpe.response.text)
            ) from httpe
        except requests.JSONDecodeError as je:
            raise LlmClientError(f"LLM returned invalid JSON: {je}") from je
        except requests.RequestException as re:
            raise LlmClientError(f"Could not reach the LLM: {re}") from re

    def stream(self, prompt: str | Messages, **additional_kwargs) -> Generator[str]:
        """
        Call the LLM and yield the response as it arrives.

        Parameters
        ----------
        prompt : str | Messages
            The prompt as a string or a list of messages.
        **additional_kwargs
            Extra fields added to the request payload, e.g. ``temperature``.

        Yields
        ------
        str
            One non-empty line of the streamed response.

        Raises
        ------
        ValueError
            If the prompt is invalid.
        ContextWindowExceededError
            If the prompt would not fit the context window. Nothing is sent.
        LlmClientError
            If the LLM cannot be reached, times out or returns an error status.
        """
        invoke_payload = self.__get_llm_payload(prompt, stream=True, **additional_kwargs)
        try:
            with requests.post(
                url=self.base_url,
                json=invoke_payload,
                headers={**self.headers, "Accept": "text/event-stream"},
                stream=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                # Server-Sent Events are always UTF-8
                response.encoding = "utf-8"
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        yield line
        except requests.HTTPError as httpe:
            raise LlmClientError(
                _http_error_message(httpe.response.status_code, httpe.response.text)
            ) from httpe
        except requests.RequestException as re:
            raise LlmClientError(f"Could not reach the LLM: {re}") from re

    async def ainvoke(self, prompt: str | Messages, **additional_kwargs) -> dict:
        """
        Call the LLM asynchronously and return the full response.

        Parameters
        ----------
        prompt : str | Messages
            The prompt as a string or a list of messages.
        **additional_kwargs
            Extra fields added to the request payload, e.g. ``temperature``.

        Returns
        -------
        dict
            The JSON response from the LLM.

        Raises
        ------
        ValueError
            If the prompt is invalid.
        ContextWindowExceededError
            If the prompt would not fit the context window. Nothing is sent.
        LlmClientError
            If the LLM cannot be reached, times out, returns an error status
            or returns a response that is not valid JSON.
        """
        invoke_payload = self.__get_llm_payload(prompt, **additional_kwargs)
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url=self.base_url,
                    json=invoke_payload,
                    headers={**self.headers, "Accept": "application/json"},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as httpe:
            raise LlmClientError(
                _http_error_message(httpe.response.status_code, httpe.response.text)
            ) from httpe
        except httpx.HTTPError as httpe:
            raise LlmClientError(f"Could not reach the LLM: {httpe!r}") from httpe
        except ValueError as je:
            raise LlmClientError(f"LLM returned invalid JSON: {je}") from je

    async def astream(
        self, prompt: str | Messages, **additional_kwargs
    ) -> AsyncGenerator[str]:
        """
        Call the LLM asynchronously and yield the response as it arrives.

        Parameters
        ----------
        prompt : str | Messages
            The prompt as a string or a list of messages.
        **additional_kwargs
            Extra fields added to the request payload, e.g. ``temperature``.

        Yields
        ------
        str
            One non-empty line of the streamed response.

        Raises
        ------
        ValueError
            If the prompt is invalid.
        ContextWindowExceededError
            If the prompt would not fit the context window. Nothing is sent.
        LlmClientError
            If the LLM cannot be reached, times out or returns an error status.
        """
        invoke_payload = self.__get_llm_payload(prompt, stream=True, **additional_kwargs)
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                async with client.stream(
                    "POST",
                    url=self.base_url,
                    json=invoke_payload,
                    headers={**self.headers, "Accept": "text/event-stream"},
                ) as response:
                    if response.is_error:
                        await response.aread()
                        raise LlmClientError(
                            _http_error_message(response.status_code, response.text)
                        )
                    async for line in response.aiter_lines():
                        if line:
                            yield line
        except httpx.HTTPError as httpe:
            raise LlmClientError(f"Could not reach the LLM: {httpe!r}") from httpe
