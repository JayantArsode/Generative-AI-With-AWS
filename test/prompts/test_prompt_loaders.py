import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.prompts import prompt_loaders
from app.prompts.prompt_loaders import SystemPrompts, get_system_prompt

load_yml_as_dict = getattr(prompt_loaders, "__load_yml_as_dict")


class PromptFileTestCase(unittest.TestCase):
    """
    Base test case that writes prompt files to a temporary folder.
    """

    def setUp(self):
        load_yml_as_dict.cache_clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(load_yml_as_dict.cache_clear)

    def write_prompt_file(self, text):
        path = Path(self.temp_dir.name) / "system_prompts.yml"
        path.write_text(text, encoding="utf-8")
        return path


class TestLoadYmlAsDict(PromptFileTestCase):
    """
    Tests for ``__load_yml_as_dict``.
    """

    def test_loads_key_value_pairs(self):
        path = self.write_prompt_file("A: hello\nB: world\n")

        self.assertEqual(load_yml_as_dict(path), {"A": "hello", "B": "world"})

    def test_missing_file_raises_file_not_found(self):
        missing = Path(self.temp_dir.name) / "missing.yml"

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_yml_as_dict(missing)

    def test_invalid_yaml_raises_value_error(self):
        path = self.write_prompt_file("A: [unclosed\n")

        with self.assertRaisesRegex(ValueError, "Error parsing YAML"):
            load_yml_as_dict(path)

    def test_non_mapping_raises_value_error(self):
        path = self.write_prompt_file("- just\n- a list\n")

        with self.assertRaisesRegex(ValueError, "key: value"):
            load_yml_as_dict(path)


class TestGetSystemPrompt(PromptFileTestCase):
    """
    Tests for ``get_system_prompt``.
    """

    def test_real_default_prompt_exists(self):
        prompt = get_system_prompt(SystemPrompts.DEFAULT_PROMPT)

        self.assertIsInstance(prompt, str)
        self.assertTrue(prompt)

    def test_returns_stripped_prompt(self):
        path = self.write_prompt_file("DEFAULT_SYSTEM_PROMPT: |\n  Be helpful.\n\n")

        with patch.object(prompt_loaders, "SYSTEM_PROMPTS_FILE", path):
            prompt = get_system_prompt(SystemPrompts.DEFAULT_PROMPT)

        self.assertEqual(prompt, "Be helpful.")

    def test_missing_key_raises_key_error(self):
        path = self.write_prompt_file("OTHER_PROMPT: hi\n")

        with patch.object(prompt_loaders, "SYSTEM_PROMPTS_FILE", path):
            with self.assertRaises(KeyError):
                get_system_prompt(SystemPrompts.DEFAULT_PROMPT)

    def test_empty_prompt_raises_key_error(self):
        path = self.write_prompt_file('DEFAULT_SYSTEM_PROMPT: "   "\n')

        with patch.object(prompt_loaders, "SYSTEM_PROMPTS_FILE", path):
            with self.assertRaises(KeyError):
                get_system_prompt(SystemPrompts.DEFAULT_PROMPT)
