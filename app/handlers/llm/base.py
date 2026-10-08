from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator
from pydantic import ValidationError
from app.schemas.prompt_templates import Messages


class BaseLlmHandler(ABC):
    """
    Base class for LLM clients.

    Subclasses must implement sync and async calls, with and without streaming.
    """

    @abstractmethod
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
            The LLM response.
        """
        ...

    @abstractmethod
    def stream(self, prompt: str | Messages, **additional_kwargs) -> Generator[dict]:
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
        dict
            One streamed JSON chunk, e.g.
            ``{"choices": [{"delta": {"content": "Hel"}}]}``.
        """
        ...

    @abstractmethod
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
            The LLM response.
        """
        ...

    @abstractmethod
    def astream(
        self, prompt: str | Messages, **additional_kwargs
    ) -> AsyncGenerator[dict]:
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
        dict
            One streamed JSON chunk, e.g.
            ``{"choices": [{"delta": {"content": "Hel"}}]}``.
        """
        ...

    @staticmethod
    def to_message(prompt: str | list | Messages) -> list[dict]:
        """
        Take a prompt as input and convert it into the list of messages
        expected by the LLM.

        Parameters
        ----------
        prompt : str | list | Messages
            The prompt provided by the user. It can be a string, a list of
            system, user or assistant messages, or a ``Messages`` object.

        Returns
        -------
        list[dict]
            The prompt converted into the format expected by the LLM:

            [
                {
                    "role": <role>,       # system | user | assistant
                    "content": <content>  # text content
                },
                ...
            ]

        Raises
        ------
        ValueError:
            If the prompt is invalid in any sense.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        if isinstance(prompt, Messages):
            messages = prompt
        elif isinstance(prompt, list):
            try:
                messages = Messages(messages=prompt)
            except ValidationError as ve:
                raise ValueError(f"Invalid prompt message found: {ve}")
        else:
            raise ValueError("Invalid prompt it should either be str, list or Messages.")

        return messages.model_dump(mode="python")["messages"]
