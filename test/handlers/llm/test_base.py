import unittest
from unittest.mock import patch

from app.handlers.llm.base import BaseLlmHandler
from app.schemas import prompt_templates
from app.schemas.prompt_templates import Messages


class TestToMessage(unittest.TestCase):
    """
    Tests for ``BaseLlmHandler.to_message``.
    """

    def setUp(self):
        patcher = patch.object(
            prompt_templates, "get_system_prompt", return_value="Default prompt"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_string_becomes_system_and_user_messages(self):
        self.assertEqual(
            BaseLlmHandler.to_message("hi"),
            [
                {"role": "system", "content": "Default prompt"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_list_of_dicts_is_converted(self):
        prompt = [
            {"role": "system", "content": "Be short."},
            {"role": "user", "content": "hi"},
        ]

        self.assertEqual(BaseLlmHandler.to_message(prompt), prompt)

    def test_messages_object_is_converted(self):
        messages = Messages(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(
            BaseLlmHandler.to_message(messages)[-1], {"role": "user", "content": "hi"}
        )

    def test_invalid_list_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid prompt message found"):
            BaseLlmHandler.to_message([{"role": "assistant", "content": "x"}])

    def test_wrong_type_raises_value_error(self):
        for prompt in (42, None, {"role": "user", "content": "hi"}):
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    BaseLlmHandler.to_message(prompt)


class TestBaseLlmHandler(unittest.TestCase):
    """
    Tests for the ``BaseLlmHandler`` abstract class.
    """

    def test_cannot_be_created_directly(self):
        with self.assertRaises(TypeError):
            BaseLlmHandler()

    def test_subclass_must_implement_all_methods(self):
        class OnlyInvoke(BaseLlmHandler):
            def invoke(self, prompt, **additional_kwargs):
                return {}

        with self.assertRaises(TypeError):
            OnlyInvoke()
