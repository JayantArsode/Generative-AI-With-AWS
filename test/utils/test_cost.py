import unittest
from decimal import Decimal

import yaml

from app.schemas.llm_response import TokenUsage
from app.schemas.provider_config import TokenPrices
from app.utils.cost import calculate_cost

# The test data from challenge 1.1
CHALLENGE_USAGE = TokenUsage(input_tokens=1000, cached_input_tokens=600, output_tokens=200)
CHALLENGE_PRICES = {"input": "1.00", "cached_input": "0.10", "output": "4.00"}


class TestCalculateCost(unittest.TestCase):
    """
    Tests for ``calculate_cost``.
    """

    def test_challenge_test_data_is_exactly_0_00126(self):
        cost = calculate_cost(CHALLENGE_USAGE, TokenPrices(**CHALLENGE_PRICES))

        self.assertEqual(cost, Decimal("0.00126"))

    def test_exact_with_prices_read_from_yaml_floats(self):
        # YAML reads 0.10 as a float, which is not exact in binary.
        prices = yaml.safe_load("{input: 1.00, cached_input: 0.10, output: 4.00}")

        cost = calculate_cost(CHALLENGE_USAGE, TokenPrices(**prices))

        self.assertEqual(cost, Decimal("0.00126"))

    def test_each_part_has_its_own_price(self):
        prices = TokenPrices(**CHALLENGE_PRICES)
        cases = {
            "uncached input": (TokenUsage(input_tokens=400, output_tokens=0), "0.0004"),
            "cached input": (
                TokenUsage(input_tokens=600, cached_input_tokens=600, output_tokens=0),
                "0.00006",
            ),
            "output": (TokenUsage(input_tokens=0, output_tokens=200), "0.0008"),
        }
        for name, (usage, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(calculate_cost(usage, prices), Decimal(expected))

    def test_no_cache_price_charges_cached_tokens_as_input(self):
        prices = TokenPrices(input="1.00", output="4.00")

        cost = calculate_cost(CHALLENGE_USAGE, prices)

        # 1,000 input x $1 + 200 output x $4 = $1,800 per million
        self.assertEqual(cost, Decimal("0.0018"))

    def test_free_model_costs_nothing(self):
        prices = TokenPrices(input=0, cached_input=0, output=0)

        self.assertEqual(calculate_cost(CHALLENGE_USAGE, prices), Decimal(0))
