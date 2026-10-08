import io
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

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


def make_answer(text, streamed=False):
    return LlmAnswer(
        text=text,
        provider="test",
        model="test-model",
        usage=TokenUsage(input_tokens=10, output_tokens=2),
        latency_seconds=0.5,
        cost=Decimal("0.000018"),
        time_to_first_token_seconds=0.25 if streamed else None,
        tokens_per_second=40.0 if streamed else None,
    )


def fake_stream(*replies):
    """
    Build a stand-in for ``call_agent_with_stream``.

    Each call streams the next reply: a list of text pieces followed by the
    final ``LlmAnswer``, or an exception to raise after the pieces.

    Returns
    -------
    MagicMock
        Records how it was called, and returns a new async generator each time.
    """
    scripts = iter(replies)

    def call(**kwargs):
        pieces, end = next(scripts)

        async def generate():
            for piece in pieces:
                yield piece
            if isinstance(end, Exception):
                raise end
            yield end

        return generate()

    return MagicMock(side_effect=call)


class CliTestCase(unittest.TestCase):
    """
    Base test case that captures stdout and stderr and fakes the agent.
    """

    def setUp(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.agent = AsyncMock()
        self.stream_agent = fake_stream()
        for patcher in (
            patch("sys.stdout", self.stdout),
            patch("sys.stderr", self.stderr),
            patch.object(main_module, "call_agent", self.agent),
            patch.object(main_module, "get_provider", return_value=PROVIDER),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def use_stream_replies(self, *replies):
        self.stream_agent = fake_stream(*replies)
        patcher = patch.object(
            main_module, "call_agent_with_stream", self.stream_agent
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, argv, user_inputs=()):
        with patch("builtins.input", side_effect=[*user_inputs, EOFError]):
            return main_module.run(argv)


class TestAsk(CliTestCase):
    """
    Tests for ``cirrus ask``.
    """

    def test_streams_answer_then_prints_stats(self):
        self.use_stream_replies((["Pa", "ris"], make_answer("Paris", streamed=True)))

        exit_code = self.run_cli(["ask", "Capital of France?"])

        self.assertEqual(exit_code, 0)
        self.stream_agent.assert_called_once_with(
            provider=PROVIDER, user_query="Capital of France?", use_history=False
        )
        self.agent.assert_not_awaited()
        self.assertEqual(self.stdout.getvalue(), "Paris\n")
        self.assertIn(
            "in 10 tok | out 2 tok | 0.50s | TTFT 0.25s | 40.0 tok/s | $0.000018",
            self.stderr.getvalue(),
        )

    def test_no_stream_waits_for_the_whole_answer(self):
        self.agent.return_value = make_answer("Paris")

        exit_code = self.run_cli(["ask", "Capital of France?", "--no-stream"])

        self.assertEqual(exit_code, 0)
        self.agent.assert_awaited_once_with(
            provider=PROVIDER, user_query="Capital of France?", use_history=False
        )
        self.assertEqual(self.stdout.getvalue(), "Paris\n")
        self.assertIn(
            "in 10 tok | out 2 tok | 0.50s | $0.000018", self.stderr.getvalue()
        )
        self.assertNotIn("TTFT", self.stderr.getvalue())

    def test_reads_question_from_stdin(self):
        self.use_stream_replies((["ok"], make_answer("ok", streamed=True)))

        with patch("sys.stdin", io.StringIO("long question from a file")):
            self.run_cli(["ask", "-"])

        self.assertEqual(
            self.stream_agent.call_args.kwargs["user_query"],
            "long question from a file",
        )

    def test_uses_chosen_provider(self):
        self.use_stream_replies((["ok"], make_answer("ok", streamed=True)))

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
                self.use_stream_replies(([], error))

                exit_code = self.run_cli(["ask", "hi"])

                self.assertEqual(exit_code, 1)
                self.assertEqual(self.stderr.getvalue(), f"Error: {error}\n")

    def test_error_mid_stream_starts_on_a_new_line(self):
        self.use_stream_replies((["Par"], LlmClientError("connection dropped")))

        exit_code = self.run_cli(["ask", "hi"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(self.stdout.getvalue(), "Par\n")
        self.assertEqual(self.stderr.getvalue(), "Error: connection dropped\n")

    def test_bad_provider_config_stops_before_calling(self):
        main_module.get_provider.side_effect = ProviderConfigError(
            "Unknown provider 'x'"
        )
        self.use_stream_replies()

        exit_code = self.run_cli(["ask", "hi", "--provider", "x"])

        self.assertEqual(exit_code, 1)
        self.stream_agent.assert_not_called()
        self.assertIn("Error: Unknown provider 'x'", self.stderr.getvalue())


class TestChat(CliTestCase):
    """
    Tests for ``cirrus chat``.
    """

    def test_streams_with_history_on_by_default(self):
        self.use_stream_replies(
            (["Pa", "ris"], make_answer("Paris", streamed=True)),
            (["2 ", "million"], make_answer("2 million", streamed=True)),
        )

        exit_code = self.run_cli(["chat"], ["Capital of France?", "Its population?"])

        self.assertEqual(exit_code, 0)
        for call in self.stream_agent.call_args_list:
            self.assertTrue(call.kwargs["use_history"])
        self.assertEqual(self.stdout.getvalue(), "> Paris\n> 2 million\n")
        self.assertIn("history on, streaming", self.stderr.getvalue())

    def test_no_history_flag(self):
        self.use_stream_replies((["ok"], make_answer("ok", streamed=True)))

        self.run_cli(["chat", "--no-history"], ["hi"])

        self.assertFalse(self.stream_agent.call_args.kwargs["use_history"])
        self.assertIn("history off", self.stderr.getvalue())

    def test_no_stream_flag(self):
        self.agent.return_value = make_answer("Paris")

        self.run_cli(["chat", "--no-stream"], ["Capital of France?"])

        self.agent.assert_awaited_once()
        self.assertEqual(self.stdout.getvalue(), "> Paris\n")
        self.assertIn("not streaming", self.stderr.getvalue())

    def test_skips_empty_input(self):
        self.use_stream_replies()

        self.run_cli(["chat"], ["", "   "])

        self.stream_agent.assert_not_called()

    def test_error_is_printed_and_chat_continues(self):
        self.use_stream_replies(
            ([], LlmClientError("HTTP 503: busy")),
            (["Paris"], make_answer("Paris", streamed=True)),
        )

        self.run_cli(["chat"], ["q1", "q2"])

        self.assertEqual(self.stream_agent.call_count, 2)
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

    def test_stream_is_on_by_default(self):
        args = main_module.build_parser().parse_args(["ask", "hi"])

        self.assertFalse(args.no_stream)
