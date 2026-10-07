from decimal import Decimal
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, model_validator


class TokenUsage(BaseModel):
    """
    Tokens used by one LLM call.

    Attributes
    ----------
    input_tokens : int
        All input tokens, cached ones included.
    cached_input_tokens : int
        The part of ``input_tokens`` that the provider served from its cache.
    output_tokens : int
        Tokens in the answer.
    estimated : bool
        ``True`` if the provider did not report usage and the numbers were
        counted locally with ``tiktoken``.
    """

    input_tokens: NonNegativeInt
    cached_input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt
    estimated: bool = False

    @model_validator(mode="after")
    def check_cached_tokens(self) -> "TokenUsage":
        """
        Make sure cached tokens are not more than all input tokens.

        Returns
        -------
        TokenUsage
            The validated usage.

        Raises
        ------
        ValueError
            If ``cached_input_tokens`` is bigger than ``input_tokens``.
        """
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot be more than input_tokens")
        return self


class LlmAnswer(BaseModel):
    """
    An LLM reply together with what it took to get it.

    Attributes
    ----------
    text : str
        The reply from the model.
    provider : str
        Name of the provider that answered.
    model : str
        Name of the model that answered.
    usage : TokenUsage
        Tokens used by the call.
    latency_seconds : float
        Time from sending the request to getting the full reply.
    cost : Decimal
        Cost of the call in US dollars.
    """

    text: str
    provider: str
    model: str
    usage: TokenUsage
    latency_seconds: NonNegativeFloat
    cost: Decimal
