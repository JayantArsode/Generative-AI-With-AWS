import unittest
from decimal import Decimal

from pydantic import ValidationError

from app.schemas.llm_response import LlmAnswer, TokenUsage


class TestTokenUsage(unittest.TestCase):
    """
    Tests for ``TokenUsage``.
    """

    def test_defaults(self):
        usage = TokenUsage(input_tokens=10, output_tokens=2)

        self.assertEqual(usage.cached_input_tokens, 0)
        self.assertFalse(usage.estimated)

    def test_cached_cannot_exceed_input(self):
        with self.assertRaisesRegex(ValidationError, "cannot be more than input_tokens"):
            TokenUsage(input_tokens=10, cached_input_tokens=11, output_tokens=0)

    def test_negative_counts_are_rejected(self):
        with self.assertRaises(ValidationError):
            TokenUsage(input_tokens=-1, output_tokens=0)


class TestLlmAnswer(unittest.TestCase):
    """
    Tests for ``LlmAnswer``.
    """

    def test_holds_reply_and_stats(self):
        answer = LlmAnswer(
            text="Paris",
            provider="groq",
            model="llama",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            latency_seconds=0.5,
            cost=Decimal("0.000018"),
        )

        self.assertEqual(answer.text, "Paris")
        self.assertEqual(answer.cost, Decimal("0.000018"))

    def test_latency_cannot_be_negative(self):
        with self.assertRaises(ValidationError):
            LlmAnswer(
                text="x",
                provider="p",
                model="m",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                latency_seconds=-1,
                cost=Decimal(0),
            )

    def test_streaming_stats_are_optional(self):
        answer = LlmAnswer(
            text="x",
            provider="p",
            model="m",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            latency_seconds=1,
            cost=Decimal(0),
        )

        self.assertIsNone(answer.time_to_first_token_seconds)
        self.assertIsNone(answer.tokens_per_second)

    def test_streaming_stats_cannot_be_negative(self):
        for field in ("time_to_first_token_seconds", "tokens_per_second"):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    LlmAnswer(
                        text="x",
                        provider="p",
                        model="m",
                        usage=TokenUsage(input_tokens=1, output_tokens=1),
                        latency_seconds=1,
                        cost=Decimal(0),
                        **{field: -1},
                    )
