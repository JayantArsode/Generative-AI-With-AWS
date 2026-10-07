from decimal import Decimal
from app.schemas.llm_response import TokenUsage
from app.schemas.provider_config import TokenPrices

TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)


def calculate_cost(usage: TokenUsage, prices: TokenPrices) -> Decimal:
    """
    Calculate the exact cost of one LLM call in US dollars.

    Cached tokens are part of the input count, so they are taken out of the
    input and charged at the cached price. Uncached input, cached input and
    output are each charged at their own price.

    Parameters
    ----------
    usage : TokenUsage
        Tokens used by the call.
    prices : TokenPrices
        Prices per million tokens.

    Returns
    -------
    Decimal
        The cost in US dollars, with no rounding.

    Examples
    --------
    >>> usage = TokenUsage(input_tokens=1000, cached_input_tokens=600, output_tokens=200)
    >>> prices = TokenPrices(input="1.00", cached_input="0.10", output="4.00")
    >>> calculate_cost(usage, prices) == Decimal("0.00126")
    True
    """
    cached_price = (
        prices.cached_input if prices.cached_input is not None else prices.input
    )
    uncached_input_tokens = usage.input_tokens - usage.cached_input_tokens

    total = (
        uncached_input_tokens * prices.input
        + usage.cached_input_tokens * cached_price
        + usage.output_tokens * prices.output
    )
    return total / TOKENS_PER_PRICE_UNIT
