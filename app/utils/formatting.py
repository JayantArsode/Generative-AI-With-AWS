from decimal import Decimal
from app.schemas.llm_response import LlmAnswer


def format_cost(cost: Decimal) -> str:
    """
    Format a cost in US dollars without hiding tiny amounts.

    Parameters
    ----------
    cost : Decimal
        The cost in US dollars.

    Returns
    -------
    str
        The cost with six decimals, e.g. ``$0.001260``.
    """
    return f"${cost:.6f}"


def format_answer_stats(answer: LlmAnswer) -> str:
    """
    Build the one-line summary printed after each answer.

    Parameters
    ----------
    answer : LlmAnswer
        The answer to summarise.

    Returns
    -------
    str
        Input tokens, output tokens, latency, time to first token and
        tokens per second (when streamed), cost and model, e.g.
        ``in 22 tok | out 87 tok | 1.23s | TTFT 0.41s | 52.3 tok/s | $0.000000 | model via nvidia``.
        Token counts start with ``~`` when they were estimated locally.
    """
    usage = answer.usage
    approx = "~" if usage.estimated else ""

    input_part = f"in {approx}{usage.input_tokens:,} tok"
    if usage.cached_input_tokens:
        input_part += f" ({usage.cached_input_tokens:,} cached)"

    parts = [
        input_part,
        f"out {approx}{usage.output_tokens:,} tok",
        f"{answer.latency_seconds:.2f}s",
    ]
    if answer.time_to_first_token_seconds is not None:
        parts.append(f"TTFT {answer.time_to_first_token_seconds:.2f}s")
    if answer.tokens_per_second is not None:
        parts.append(f"{answer.tokens_per_second:.1f} tok/s")
    parts += [
        format_cost(answer.cost),
        f"{answer.model} via {answer.provider}",
    ]
    if usage.estimated:
        parts.append("tokens estimated")
    return "  " + " | ".join(parts)
