import unittest

from app.exceptions import ProviderConfigError


class TestProviderConfigError(unittest.TestCase):
    """
    Tests for ``app.exceptions.ProviderConfigError``.
    """

    def test_is_an_exception_with_message(self):
        error = ProviderConfigError("Set GROQ_API_KEY in .env")

        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Set GROQ_API_KEY in .env")
