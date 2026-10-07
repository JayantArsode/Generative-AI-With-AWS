from decimal import Decimal
from pydantic import BaseModel, NonNegativeInt, PositiveInt, field_validator, model_validator


class TokenPrices(BaseModel):
    """
    Prices of a model in US dollars per million tokens.

    Attributes
    ----------
    input : Decimal
        Price of uncached input tokens.
    cached_input : Decimal | None
        Price of cached input tokens. ``None`` means the provider has no
        cache discount, so cached tokens cost the same as ``input``.
    output : Decimal
        Price of output tokens.
    """

    input: Decimal
    cached_input: Decimal | None = None
    output: Decimal

    @field_validator("input", "cached_input", "output", mode="before")
    @classmethod
    def float_to_exact_decimal(cls, value):
        """
        Turn floats from YAML into exact decimals.

        ``Decimal(0.1)`` is ``0.1000000000000000055...``, but
        ``Decimal("0.1")`` is exactly ``0.1``. Going through ``str`` keeps
        prices exact, so costs are exact too.

        Parameters
        ----------
        value : Any
            The raw value from the config file.

        Returns
        -------
        Any
            The value as a string if it was a float, otherwise unchanged.
        """
        if isinstance(value, float):
            return str(value)
        return value

    @field_validator("input", "cached_input", "output")
    @classmethod
    def not_negative(cls, value: Decimal | None) -> Decimal | None:
        """
        Make sure no price is negative.

        Parameters
        ----------
        value : Decimal | None
            The price to check.

        Returns
        -------
        Decimal | None
            The same price.

        Raises
        ------
        ValueError
            If the price is below zero.
        """
        if value is not None and value < 0:
            raise ValueError("prices cannot be negative")
        return value


class ProviderConfig(BaseModel):
    """
    Settings for one OpenAI-compatible provider and model.

    Attributes
    ----------
    name : str
        Name of the provider, taken from its key in the config file.
    base_url : str
        Full URL of the provider's chat completions endpoint.
    api_key_env : str | None
        Name of the environment variable that holds the API key.
        ``None`` for providers that need no key, such as a local Ollama.
    model : str
        Name of the model to call.
    context_window : int
        Maximum number of tokens the model accepts, input and output together.
    reserved_output_tokens : int
        Tokens kept free in the context window for the answer.
    prices : TokenPrices
        Prices per million tokens.
    """

    name: str = ""
    base_url: str
    api_key_env: str | None = None
    model: str
    context_window: PositiveInt
    reserved_output_tokens: NonNegativeInt = 1024
    prices: TokenPrices

    @model_validator(mode="after")
    def check_room_for_input(self) -> "ProviderConfig":
        """
        Make sure the reserved output tokens leave room for some input.

        Returns
        -------
        ProviderConfig
            The validated provider.

        Raises
        ------
        ValueError
            If ``reserved_output_tokens`` uses up the whole context window.
        """
        if self.reserved_output_tokens >= self.context_window:
            raise ValueError(
                "reserved_output_tokens must be smaller than context_window"
            )
        return self

    @property
    def max_input_tokens(self) -> int:
        """
        Tokens left for the input after reserving room for the answer.

        Returns
        -------
        int
            ``context_window`` minus ``reserved_output_tokens``.
        """
        return self.context_window - self.reserved_output_tokens


class ProvidersConfig(BaseModel):
    """
    The whole providers config file.

    Attributes
    ----------
    default : str
        Name of the provider used when none is given.
    providers : dict[str, ProviderConfig]
        All providers, by name.
    """

    default: str
    providers: dict[str, ProviderConfig]

    @model_validator(mode="after")
    def check_providers(self) -> "ProvidersConfig":
        """
        Fill in provider names and make sure the default exists.

        Returns
        -------
        ProvidersConfig
            The validated config.

        Raises
        ------
        ValueError
            If the default provider is not in ``providers``.
        """
        for name, provider in self.providers.items():
            provider.name = name
        if self.default not in self.providers:
            raise ValueError(
                f"default provider '{self.default}' is not in providers: "
                f"{', '.join(self.providers) or 'none defined'}"
            )
        return self
