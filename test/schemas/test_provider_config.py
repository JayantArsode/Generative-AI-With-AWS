import unittest
from decimal import Decimal

from pydantic import ValidationError

from app.schemas.provider_config import ProviderConfig, ProvidersConfig, TokenPrices


def provider_data(**overrides):
    data = {
        "base_url": "https://llm.example/v1/chat/completions",
        "api_key_env": "TEST_API_KEY",
        "model": "m",
        "context_window": 8000,
        "prices": {"input": 1, "output": 4},
    }
    return {**data, **overrides}


class TestTokenPrices(unittest.TestCase):
    """
    Tests for ``TokenPrices``.
    """

    def test_floats_become_exact_decimals(self):
        prices = TokenPrices(input=1.0, cached_input=0.1, output=4.0)

        self.assertEqual(prices.cached_input, Decimal("0.1"))

    def test_cached_price_is_optional(self):
        self.assertIsNone(TokenPrices(input=1, output=4).cached_input)

    def test_negative_price_is_rejected(self):
        with self.assertRaises(ValidationError):
            TokenPrices(input=-1, output=4)


class TestProviderConfig(unittest.TestCase):
    """
    Tests for ``ProviderConfig``.
    """

    def test_defaults(self):
        provider = ProviderConfig(**provider_data())

        self.assertEqual(provider.reserved_output_tokens, 1024)
        self.assertEqual(provider.max_input_tokens, 8000 - 1024)

    def test_api_key_env_can_be_empty(self):
        provider = ProviderConfig(**provider_data(api_key_env=None))

        self.assertIsNone(provider.api_key_env)

    def test_context_window_must_be_positive(self):
        with self.assertRaises(ValidationError):
            ProviderConfig(**provider_data(context_window=0))

    def test_reserved_output_must_leave_room_for_input(self):
        with self.assertRaisesRegex(ValidationError, "smaller than context_window"):
            ProviderConfig(**provider_data(context_window=1000, reserved_output_tokens=1000))

    def test_required_fields(self):
        for field in ("base_url", "model", "context_window", "prices"):
            with self.subTest(field=field):
                data = provider_data()
                del data[field]
                with self.assertRaises(ValidationError):
                    ProviderConfig(**data)


class TestProvidersConfig(unittest.TestCase):
    """
    Tests for ``ProvidersConfig``.
    """

    def test_names_come_from_keys(self):
        config = ProvidersConfig(
            default="a", providers={"a": provider_data(), "b": provider_data()}
        )

        self.assertEqual(config.providers["a"].name, "a")
        self.assertEqual(config.providers["b"].name, "b")

    def test_default_must_exist(self):
        with self.assertRaisesRegex(ValidationError, "default provider 'missing'"):
            ProvidersConfig(default="missing", providers={"a": provider_data()})
