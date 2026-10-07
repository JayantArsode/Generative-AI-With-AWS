import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.schemas import prompt_templates
from app.schemas.prompt_templates import (
    AssistantMessage,
    Messages,
    SystemMessage,
    UserMessage,
)


class TestMessageModels(unittest.TestCase):
    """
    Tests for the single message models.
    """

    def test_roles_are_set_by_default(self):
        self.assertEqual(SystemMessage(content="x").role, "system")
        self.assertEqual(UserMessage(content="x").role, "user")
        self.assertEqual(AssistantMessage(content="x").role, "assistant")

    def test_content_is_required(self):
        with self.assertRaises(ValidationError):
            UserMessage()


class TestMessages(unittest.TestCase):
    """
    Tests for ``Messages`` and its validation rules.
    """

    def setUp(self):
        patcher = patch.object(
            prompt_templates, "get_system_prompt", return_value="Default prompt"
        )
        self.mock_get_system_prompt = patcher.start()
        self.addCleanup(patcher.stop)

    def dump(self, messages):
        return Messages(messages=messages).model_dump()["messages"]

    def test_adds_default_system_prompt_when_missing(self):
        messages = self.dump([{"role": "user", "content": "hi"}])

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Default prompt"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_keeps_own_system_prompt(self):
        prompt = [
            {"role": "system", "content": "Be short."},
            {"role": "user", "content": "hi"},
        ]

        self.assertEqual(self.dump(prompt), prompt)
        self.mock_get_system_prompt.assert_not_called()

    def test_history_starting_with_assistant_is_allowed(self):
        messages = self.dump(
            [
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "Follow-up"},
            ]
        )

        self.assertEqual(
            [m["role"] for m in messages], ["system", "assistant", "user"]
        )

    def test_invalid_conversations_raise(self):
        cases = {
            "system not first": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "late"},
                {"role": "user", "content": "hi again"},
            ],
            "no user message": [{"role": "system", "content": "rules"}],
            "last message not user": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "unknown role": [{"role": "robot", "content": "hi"}],
        }
        for name, messages in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValidationError):
                    Messages(messages=messages)
