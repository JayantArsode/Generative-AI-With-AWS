import io
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from app import main as main_module
from app.exceptions import (
    ContextWindowExceededError,
    LlmClientError,
    ProviderConfigError,
)
from app.schemas.llm_response import LlmAnswer, TokenUsage
from app.schemas.provider_config import ProviderConfig

PROVIDER = ProviderConfig(
    name="test",
    base_url="https://llm.example/v1/chat/completions",
    api_key_env="TEST_API_KEY",
    model="test-model",
    context_window=8000,
    prices={"input": 1, "output": 4},
)


def make_answer(text):
    return LlmAnswer(
        text=text,
        provider="test",
        model="test-model",
        usage=TokenUsage(input_tokens=10, output_tokens=2),
        latency_seconds=0.5,
        cost=Decimal("0.000018"),
    )


class CliTestCase(unittest.TestCase):
    """
    Base test case that captures stdout and stderr and fakes the agent.
    """

    def setUp(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.agent = AsyncMock()
        for patcher in (
            patch("sys.stdout", self.stdout),
            patch("sys.stderr", self.stderr),
            patch.object(main_module, "call_agent_with_history", self.agent),
            patch.object(main_module, "get_provider", return_value=PROVIDER),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_cli(self, argv, user_inputs=()):
        with patch("builtins.input", side_effect=[*user_inputs, EOFError]):
            return main_module.run(argv)


class TestAsk(CliTestCase):
    """
    Tests for ``cirrus ask``.
    """

    def test_prints_answer_and_stats(self):
        self.agent.return_value = make_answer("Paris")

        exit_code = self.run_cli(["ask", "Capital of France?"])

        self.assertEqual(exit_code, 0)
        self.agent.assert_awaited_once_with(
            provider=PROVIDER, user_query="Capital of France?", use_history=False
        )
        self.assertEqual(self.stdout.getvalue(), "Paris\n")
        self.assertIn("in 10 tok | out 2 tok | 0.50s | $0.000018", self.stderr.getvalue())

    def test_reads_question_from_stdin(self):
        self.agent.return_value = make_answer("ok")

        with patch("sys.stdin", io.StringIO("long question from a file")):
            self.run_cli(["ask", "-"])

        self.assertEqual(
            self.agent.await_args.kwargs["user_query"], "long question from a file"
        )

    def test_uses_chosen_provider(self):
        self.agent.return_value = make_answer("ok")

        self.run_cli(["ask", "hi", "--provider", "groq"])

        main_module.get_provider.assert_called_once_with("groq")

    def test_errors_are_one_line_and_exit_1(self):
        errors = [
            LlmClientError("LLM returned HTTP 401: Invalid API key"),
            ContextWindowExceededError("Request is about 500,000 tokens"),
            ProviderConfigError("Set GROQ_API_KEY in .env"),
            ValueError("Invalid prompt"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.stderr.seek(0)
                self.stderr.truncate()
                self.agent.side_effect = error

                exit_code = self.run_cli(["ask", "hi"])

                self.assertEqual(exit_code, 1)
                self.assertEqual(self.stderr.getvalue(), f"Error: {error}\n")

    def test_bad_provider_config_stops_before_calling(self):
        main_module.get_provider.side_effect = ProviderConfigError("Unknown provider 'x'")

        exit_code = self.run_cli(["ask", "hi", "--provider", "x"])

        self.assertEqual(exit_code, 1)
        self.agent.assert_not_awaited()
        self.assertIn("Error: Unknown provider 'x'", self.stderr.getvalue())


class TestChat(CliTestCase):
    """
    Tests for ``cirrus chat``.
    """

    def test_history_is_on_by_default(self):
        self.agent.side_effect = [make_answer("Paris"), make_answer("2 million")]

        exit_code = self.run_cli(["chat"], ["Capital of France?", "Its population?"])

        self.assertEqual(exit_code, 0)
        for call in self.agent.await_args_list:
            self.assertTrue(call.kwargs["use_history"])
        self.assertEqual(self.stdout.getvalue(), "> Paris\n> 2 million\n")
        self.assertIn("history on", self.stderr.getvalue())

    def test_no_history_flag(self):
        self.agent.return_value = make_answer("ok")

        self.run_cli(["chat", "--no-history"], ["hi"])

        self.assertFalse(self.agent.await_args.kwargs["use_history"])
        self.assertIn("history off", self.stderr.getvalue())

    def test_skips_empty_input(self):
        self.run_cli(["chat"], ["", "   "])

        self.agent.assert_not_awaited()

    def test_error_is_printed_and_chat_continues(self):
        self.agent.side_effect = [LlmClientError("HTTP 503: busy"), make_answer("Paris")]

        self.run_cli(["chat"], ["q1", "q2"])

        self.assertEqual(self.agent.await_count, 2)
        self.assertIn("Error: HTTP 503: busy", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "> Paris\n")

    def test_ctrl_c_exits_cleanly(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            exit_code = main_module.run(["chat"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Bye!", self.stderr.getvalue())


class TestBuildParser(unittest.TestCase):
    """
    Tests for ``build_parser``.
    """

    def test_a_command_is_required(self):
        with patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                main_module.build_parser().parse_args([])
