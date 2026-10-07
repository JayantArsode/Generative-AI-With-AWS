import unittest

from app.handlers.llm.factory import get_llm_client
from app.handlers.llm.llm_client_handlers import HTTPLlmClientHandler


class TestGetLlmClient(unittest.TestCase):
    """
    Tests for ``get_llm_client``.
    """

    def setUp(self):
        get_llm_client.cache_clear()
        self.addCleanup(get_llm_client.cache_clear)

    def test_returns_configured_http_client(self):
        client = get_llm_client(base_url="https://u", api_key="k", model="m")

        self.assertIsInstance(client, HTTPLlmClientHandler)
        self.assertEqual(client.base_url, "https://u")
        self.assertEqual(client.model, "m")
        self.assertEqual(client.headers["Authorization"], "Bearer k")

    def test_same_settings_reuse_the_same_client(self):
        first = get_llm_client(base_url="https://u", api_key="k", model="m")
        second = get_llm_client(base_url="https://u", api_key="k", model="m")

        self.assertIs(first, second)

    def test_different_settings_create_a_new_client(self):
        first = get_llm_client(base_url="https://u", api_key="k", model="m1")
        second = get_llm_client(base_url="https://u", api_key="k", model="m2")

        self.assertIsNot(first, second)

    def test_passes_context_window_settings(self):
        client = get_llm_client(
            base_url="https://u",
            api_key="k",
            model="m",
            context_window=8000,
            reserved_output_tokens=500,
        )

        self.assertEqual(client.context_window, 8000)
        self.assertEqual(client.reserved_output_tokens, 500)
