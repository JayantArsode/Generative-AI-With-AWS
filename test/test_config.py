import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import PROVIDERS_FILE, get_api_key, get_provider, load_providers
from app.exceptions import ProviderConfigError

VALID_CONFIG = """
default: first
providers:
  first:
    base_url: https://first.example/v1/chat/completions
    api_key_env: FIRST_API_KEY
    model: first-model
    context_window: 8000
    prices: {input: 1.00, cached_input: 0.10, output: 4.00}
  local:
    base_url: http://localhost:11434/v1/chat/completions
    api_key_env: null
    model: local-model
    context_window: 4000
    prices: {input: 0, output: 0}
"""


class ConfigFileTestCase(unittest.TestCase):
    """
    Base test case that writes config files to a temporary folder.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def write_config(self, text):
        path = Path(self.temp_dir.name) / "providers.yml"
        path.write_text(text, encoding="utf-8")
        return path


class TestLoadProviders(ConfigFileTestCase):
    """
    Tests for ``load_providers``.
    """

    def test_real_providers_file_is_valid(self):
        config = load_providers(PROVIDERS_FILE)

        self.assertIn(config.default, config.providers)
        self.assertGreaterEqual(len(config.providers), 2)

    def test_loads_valid_file(self):
        config = load_providers(self.write_config(VALID_CONFIG))

        self.assertEqual(config.default, "first")
        self.assertEqual(set(config.providers), {"first", "local"})

    def test_missing_file(self):
        with self.assertRaisesRegex(ProviderConfigError, "not found"):
            load_providers(Path(self.temp_dir.name) / "missing.yml")

    def test_invalid_yaml(self):
        with self.assertRaisesRegex(ProviderConfigError, "Error parsing"):
            load_providers(self.write_config("default: [unclosed\n"))

    def test_invalid_settings(self):
        path = self.write_config("default: first\nproviders:\n  first:\n    model: m\n")

        with self.assertRaisesRegex(ProviderConfigError, "Invalid providers config"):
            load_providers(path)


class TestGetProvider(ConfigFileTestCase):
    """
    Tests for ``get_provider``.
    """

    def test_default_provider(self):
        provider = get_provider(file_path=self.write_config(VALID_CONFIG))

        self.assertEqual(provider.name, "first")
        self.assertEqual(provider.model, "first-model")

    def test_named_provider(self):
        provider = get_provider("local", file_path=self.write_config(VALID_CONFIG))

        self.assertEqual(provider.name, "local")

    def test_unknown_provider_lists_choices(self):
        path = self.write_config(VALID_CONFIG)

        with self.assertRaisesRegex(ProviderConfigError, "Unknown provider 'nope'.*first, local"):
            get_provider("nope", file_path=path)


class TestGetApiKey(ConfigFileTestCase):
    """
    Tests for ``get_api_key``.
    """

    def setUp(self):
        super().setUp()
        self.path = self.write_config(VALID_CONFIG)

    def test_reads_key_from_named_variable(self):
        with patch.dict(os.environ, {"FIRST_API_KEY": "secret"}):
            key = get_api_key(get_provider("first", file_path=self.path))

        self.assertEqual(key, "secret")

    def test_missing_key_names_the_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderConfigError, "Set FIRST_API_KEY"):
                get_api_key(get_provider("first", file_path=self.path))

    def test_provider_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            key = get_api_key(get_provider("local", file_path=self.path))

        self.assertIsNone(key)
