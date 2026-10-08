import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents import cirrus_chat_agent
from app.exceptions import LlmClientError
from app.schemas.provider_config import ProviderConfig

maintain_chat_window = getattr(cirrus_chat_agent, "__maintain_chat_window")

PROVIDER = ProviderConfig(
    name="test",
    base_url="https://u",
    api_key_env="TEST_API_KEY",
    model="m",
    context_window=8000,
    reserved_output_tokens=1000,
    prices={"input": "1.00", "cached_input": "0.10", "output": "4.00"},
)


def llm_reply(text, usage=None):
    response = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        response["usage"] = usage
    return response


class TestMaintainChatWindow(unittest.TestCase):
    """
    Tests for ``__maintain_chat_window``.
    """

    def test_keeps_everything_under_budget(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        self.assertEqual(maintain_chat_window(history, max_tokens=200), history)

    def test_drops_oldest_messages_first(self):
        history = [
            {"role": "user", "content": "old " * 100},
            {"role": "assistant", "content": "older answer " * 50},
            {"role": "user", "content": "newest question"},
        ]

        trimmed = maintain_chat_window(history, max_tokens=50)

        self.assertEqual(trimmed, [{"role": "user", "content": "newest question"}])

    def test_always_keeps_newest_message(self):
        huge = {"role": "user", "content": "word " * 1000}

        self.assertEqual(maintain_chat_window([huge], max_tokens=10), [huge])

    def test_does_not_change_input_list(self):
        history = [
            {"role": "user", "content": "old " * 100},
            {"role": "user", "content": "new"},
        ]
        original = list(history)

        maintain_chat_window(history, max_tokens=20)

        self.assertEqual(history, original)


class TestCallAgent(unittest.IsolatedAsyncioTestCase):
    """
    Tests for ``call_agent``.
    """

    def setUp(self):
        cirrus_chat_agent.SESSION_CONV_HIST = []
        self.addCleanup(setattr, cirrus_chat_agent, "SESSION_CONV_HIST", [])

        self.client = MagicMock()
        self.client.ainvoke = AsyncMock()

        client_patcher = patch.object(
            cirrus_chat_agent, "get_llm_client", return_value=self.client
        )
        self.mock_get_client = client_patcher.start()
        self.addCleanup(client_patcher.stop)

        key_patcher = patch.object(cirrus_chat_agent, "get_api_key", return_value="k")
        key_patcher.start()
        self.addCleanup(key_patcher.stop)

    async def ask(self, question, use_history=True, **additional_kwargs):
        return await cirrus_chat_agent.call_agent(
            provider=PROVIDER,
            user_query=question,
            use_history=use_history,
            **additional_kwargs,
        )

    def sent_messages(self, call_index):
        return self.client.ainvoke.await_args_list[call_index].args[0]

    async def test_returns_answer_and_saves_both_turns(self):
        self.client.ainvoke.return_value = llm_reply("Paris")

        answer = await self.ask("Capital of France?")

        self.assertEqual(answer.text, "Paris")
        self.assertEqual(answer.provider, "test")
        self.assertEqual(answer.model, "m")
        self.assertGreaterEqual(answer.latency_seconds, 0)
        self.assertEqual(
            cirrus_chat_agent.SESSION_CONV_HIST,
            [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ],
        )

    async def test_builds_client_from_provider(self):
        self.client.ainvoke.return_value = llm_reply("ok")

        await self.ask("hi", temperature=0.1)

        self.mock_get_client.assert_called_once_with(
            base_url="https://u",
            api_key="k",
            model="m",
            context_window=8000,
            reserved_output_tokens=1000,
        )
        self.assertEqual(self.client.ainvoke.await_args.kwargs, {"temperature": 0.1})

    async def test_follow_up_sends_previous_turns(self):
        self.client.ainvoke.side_effect = [llm_reply("Paris"), llm_reply("2 million")]

        await self.ask("Capital of France?")
        await self.ask("Its population?")

        self.assertEqual(
            self.sent_messages(1),
            [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris"},
                {"role": "user", "content": "Its population?"},
            ],
        )

    async def test_history_off_sends_only_the_new_question(self):
        self.client.ainvoke.side_effect = [llm_reply("Paris"), llm_reply("Not sure")]

        await self.ask("Capital of France?", use_history=False)
        await self.ask("Its population?", use_history=False)

        self.assertEqual(
            self.sent_messages(1), [{"role": "user", "content": "Its population?"}]
        )
        self.assertEqual(cirrus_chat_agent.SESSION_CONV_HIST, [])

    async def test_failed_call_leaves_history_unchanged(self):
        self.client.ainvoke.side_effect = [
            llm_reply("Paris"),
            LlmClientError("HTTP 503"),
            llm_reply("2 million"),
        ]

        await self.ask("Capital of France?")
        with self.assertRaises(LlmClientError):
            await self.ask("Its population?")
        history_after_failure = list(cirrus_chat_agent.SESSION_CONV_HIST)
        await self.ask("Its population?")

        self.assertEqual(
            history_after_failure,
            [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ],
        )
        retried = [m["content"] for m in self.sent_messages(2)]
        self.assertEqual(retried.count("Its population?"), 1)

    async def test_unexpected_response_format(self):
        self.client.ainvoke.return_value = {"error": "no choices"}

        with self.assertRaisesRegex(LlmClientError, "Unexpected LLM response format"):
            await self.ask("hi")

        self.assertEqual(cirrus_chat_agent.SESSION_CONV_HIST, [])

    async def test_uses_reported_usage_for_cost(self):
        self.client.ainvoke.return_value = llm_reply(
            "ok",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
        )

        answer = await self.ask("hi")

        self.assertEqual(answer.usage.input_tokens, 1000)
        self.assertEqual(answer.usage.cached_input_tokens, 600)
        self.assertEqual(answer.usage.output_tokens, 200)
        self.assertFalse(answer.usage.estimated)
        self.assertEqual(answer.cost, Decimal("0.00126"))

    async def test_estimates_usage_when_provider_sends_none(self):
        self.client.ainvoke.return_value = llm_reply("Hello there")

        answer = await self.ask("hi")

        self.assertTrue(answer.usage.estimated)
        self.assertGreater(answer.usage.input_tokens, 0)
        self.assertGreater(answer.usage.output_tokens, 0)

    async def test_history_is_trimmed_to_fit_the_context_window(self):
        small_provider = PROVIDER.model_copy(
            update={"context_window": 300, "reserved_output_tokens": 50}
        )
        cirrus_chat_agent.SESSION_CONV_HIST = [
            {"role": "user", "content": "old " * 400},
            {"role": "assistant", "content": "old answer"},
        ]
        self.client.ainvoke.return_value = llm_reply("ok")

        await cirrus_chat_agent.call_agent(
            provider=small_provider, user_query="new question"
        )

        sent = [m["content"] for m in self.sent_messages(0)]
        self.assertNotIn("old " * 400, sent)
        self.assertEqual(sent[-1], "new question")


def stream_chunk(text=None, reasoning=None):
    delta = {}
    if text is not None:
        delta["content"] = text
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"delta": delta}], "usage": None}


USAGE_CHUNK = {
    "choices": [],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "prompt_tokens_details": {"cached_tokens": 600},
    },
}


class TestCallAgentWithStream(unittest.IsolatedAsyncioTestCase):
    """
    Tests for ``call_agent_with_stream``.
    """

    def setUp(self):
        cirrus_chat_agent.SESSION_CONV_HIST = []
        self.addCleanup(setattr, cirrus_chat_agent, "SESSION_CONV_HIST", [])

        self.client = MagicMock()
        self.streams = []
        self.sent = []

        def astream(messages, **additional_kwargs):
            self.sent.append(list(messages))
            chunks = self.streams.pop(0)

            async def generate():
                for chunk in chunks:
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk

            return generate()

        self.client.astream = MagicMock(side_effect=astream)

        for patcher in (
            patch.object(cirrus_chat_agent, "get_llm_client", return_value=self.client),
            patch.object(cirrus_chat_agent, "get_api_key", return_value="k"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def ask(self, question, use_history=True):
        pieces = []
        async for item in cirrus_chat_agent.call_agent_with_stream(
            provider=PROVIDER, user_query=question, use_history=use_history
        ):
            pieces.append(item)
        return pieces[:-1], pieces[-1]

    async def test_yields_text_pieces_then_the_full_answer(self):
        self.streams.append(
            [
                stream_chunk(reasoning="thinking..."),
                stream_chunk("Pa"),
                stream_chunk(""),
                stream_chunk("ris"),
                USAGE_CHUNK,
            ]
        )

        pieces, answer = await self.ask("Capital of France?")

        self.assertEqual(pieces, ["Pa", "ris"])
        self.assertEqual(answer.text, "Paris")
        self.assertEqual(answer.provider, "test")
        self.assertGreaterEqual(answer.latency_seconds, 0)

    async def test_reads_usage_from_the_last_chunk(self):
        self.streams.append([stream_chunk("ok"), USAGE_CHUNK])

        _, answer = await self.ask("hi")

        self.assertFalse(answer.usage.estimated)
        self.assertEqual(answer.usage.input_tokens, 1000)
        self.assertEqual(answer.usage.cached_input_tokens, 600)
        self.assertEqual(answer.cost, Decimal("0.00126"))

    async def test_estimates_usage_when_none_is_sent(self):
        self.streams.append([stream_chunk("Hello "), stream_chunk("there friend")])

        _, answer = await self.ask("hi")

        self.assertTrue(answer.usage.estimated)
        self.assertGreater(answer.usage.input_tokens, 0)
        # Estimated from the whole reply, not just the last piece
        self.assertGreaterEqual(answer.usage.output_tokens, 3)

    async def test_time_to_first_token_and_speed(self):
        self.streams.append(
            [stream_chunk("one two three four five six seven eight"), USAGE_CHUNK]
        )

        with patch.object(
            cirrus_chat_agent.time, "perf_counter", side_effect=[10.0, 10.5, 12.5]
        ):
            _, answer = await self.ask("hi")

        self.assertEqual(answer.time_to_first_token_seconds, 0.5)
        self.assertEqual(answer.latency_seconds, 2.5)
        # 8 tokens, the first one excluded, over the 2 seconds after it
        self.assertAlmostEqual(answer.tokens_per_second, 7 / 2)

    async def test_no_text_means_no_ttft_or_speed(self):
        self.streams.append([stream_chunk(reasoning="hmm"), USAGE_CHUNK])

        pieces, answer = await self.ask("hi")

        self.assertEqual(pieces, [])
        self.assertEqual(answer.text, "")
        self.assertIsNone(answer.time_to_first_token_seconds)
        self.assertIsNone(answer.tokens_per_second)

    async def test_follow_up_sends_previous_turns(self):
        self.streams += [[stream_chunk("Paris")], [stream_chunk("2 million")]]

        await self.ask("Capital of France?")
        await self.ask("Its population?")

        self.assertEqual(
            self.sent[1],
            [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris"},
                {"role": "user", "content": "Its population?"},
            ],
        )

    async def test_history_off(self):
        self.streams += [[stream_chunk("Paris")], [stream_chunk("Not sure")]]

        await self.ask("Capital of France?", use_history=False)
        await self.ask("Its population?", use_history=False)

        self.assertEqual(self.sent[1], [{"role": "user", "content": "Its population?"}])
        self.assertEqual(cirrus_chat_agent.SESSION_CONV_HIST, [])

    async def test_failure_mid_stream_leaves_history_unchanged(self):
        self.streams += [
            [stream_chunk("Paris")],
            [stream_chunk("2 mil"), LlmClientError("connection dropped")],
        ]

        await self.ask("Capital of France?")
        with self.assertRaises(LlmClientError):
            await self.ask("Its population?")

        self.assertEqual(
            cirrus_chat_agent.SESSION_CONV_HIST,
            [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ],
        )

    async def test_stopping_early_leaves_history_unchanged(self):
        self.streams.append([stream_chunk("Pa"), stream_chunk("ris")])

        stream = cirrus_chat_agent.call_agent_with_stream(
            provider=PROVIDER, user_query="Capital of France?"
        )
        self.assertEqual(await anext(stream), "Pa")
        await stream.aclose()

        self.assertEqual(cirrus_chat_agent.SESSION_CONV_HIST, [])

    async def test_streamed_error_is_raised(self):
        self.streams.append([{"error": {"message": "model overloaded"}}])

        with self.assertRaisesRegex(LlmClientError, "model overloaded"):
            await self.ask("hi")

    async def test_unexpected_chunk_is_raised(self):
        self.streams.append([{"choices": "not a list"}])

        with self.assertRaisesRegex(LlmClientError, "Unexpected LLM stream chunk"):
            await self.ask("hi")
