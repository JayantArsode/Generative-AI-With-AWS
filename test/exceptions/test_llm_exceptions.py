import unittest

from app.exceptions import ContextWindowExceededError, LlmClientError


class TestLlmClientError(unittest.TestCase):
    """
    Tests for ``app.exceptions.LlmClientError``.
    """

    def test_is_an_exception_with_message(self):
        error = LlmClientError("LLM returned HTTP 401: Invalid API key")

        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "LLM returned HTTP 401: Invalid API key")

    def test_keeps_original_cause(self):
        original = ConnectionError("connection refused")

        with self.assertRaises(LlmClientError) as ctx:
            try:
                raise original
            except ConnectionError as e:
                raise LlmClientError("Could not reach the LLM") from e

        self.assertIs(ctx.exception.__cause__, original)


class TestContextWindowExceededError(unittest.TestCase):
    """
    Tests for ``app.exceptions.ContextWindowExceededError``.
    """

    def test_is_separate_from_llm_errors(self):
        error = ContextWindowExceededError("Request is about 500,000 tokens")

        self.assertIsInstance(error, Exception)
        self.assertNotIsInstance(error, LlmClientError)
        self.assertEqual(str(error), "Request is about 500,000 tokens")
