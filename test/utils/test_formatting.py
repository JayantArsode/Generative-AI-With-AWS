import unittest
from decimal import Decimal

from app.schemas.llm_response import LlmAnswer, TokenUsage
from app.utils.formatting import format_answer_stats, format_cost


def make_answer(usage, cost="0.00126"):
    return LlmAnswer(
        text="ok",
        provider="groq",
        model="llama",
        usage=usage,
        latency_seconds=1.234,
        cost=Decimal(cost),
    )


class TestFormatCost(unittest.TestCase):
    """
    Tests for ``format_cost``.
    """

    def test_six_decimals(self):
        self.assertEqual(format_cost(Decimal("0.00126")), "$0.001260")

    def test_free(self):
        self.assertEqual(format_cost(Decimal(0)), "$0.000000")


class TestFormatAnswerStats(unittest.TestCase):
    """
    Tests for ``format_answer_stats``.
    """

    def test_reported_usage_with_cache(self):
        usage = TokenUsage(input_tokens=1000, cached_input_tokens=600, output_tokens=200)

        self.assertEqual(
            format_answer_stats(make_answer(usage)),
            "  in 1,000 tok (600 cached) | out 200 tok | 1.23s | $0.001260 | llama via groq",
        )

    def test_no_cache_hides_cached_part(self):
        usage = TokenUsage(input_tokens=10, output_tokens=2)

        self.assertNotIn("cached", format_answer_stats(make_answer(usage)))

    def test_estimated_usage_is_marked(self):
        usage = TokenUsage(input_tokens=10, output_tokens=2, estimated=True)

        stats = format_answer_stats(make_answer(usage))

        self.assertIn("in ~10 tok", stats)
        self.assertIn("out ~2 tok", stats)
        self.assertTrue(stats.endswith("tokens estimated"))
