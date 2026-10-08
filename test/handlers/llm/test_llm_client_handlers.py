import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
import requests

from app.exceptions import ContextWindowExceededError, LlmClientError
from app.handlers.llm import llm_client_handlers
from app.handlers.llm.llm_client_handlers import HTTPLlmClientHandler

REAL_ASYNC_CLIENT = httpx.AsyncClient
BASE_URL = "https://llm.example/v1/chat/completions"
OK_RESPONSE = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}

# The SSE test data from challenge 1.2, split across chunks at odd places
SSE_CHUNKS = [
    b": ping\n\n",
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
    b'\ndata: {"choices":[{"delta":{"content":"lo"}}]}\n\nda',
    b"ta: [DONE]\n\n",
]
SSE_CHUNKS_EXPECTED = [
    {"choices": [{"delta": {"content": "Hel"}}]},
    {"choices": [{"delta": {"content": "lo"}}]},
]


async def async_chunks(chunks):
    """
    Serve ``chunks`` one by one, like bytes arriving from the network.
    """
    for chunk in chunks:
        yield chunk


def make_client():
    return HTTPLlmClientHandler(base_url=BASE_URL, api_key="test-key", model="m")


def fake_requests_response(status_code, body):
    """
    Build a ``requests.Response`` without touching the network.
    """
    response = requests.Response()
    response.status_code = status_code
    response._content = body.encode()
    response._content_consumed = True
    response.url = BASE_URL
    return response


def patch_async_server(handler):
    """
    Send every ``httpx.AsyncClient`` request to ``handler`` instead of the network.
    """

    def fake_async_client(*args, **kwargs):
        return REAL_ASYNC_CLIENT(*args, transport=httpx.MockTransport(handler), **kwargs)

    return patch.object(llm_client_handlers.httpx, "AsyncClient", fake_async_client)


class TestInvoke(unittest.TestCase):
    """
    Tests for the sync ``invoke`` method.
    """

    def setUp(self):
        patcher = patch.object(llm_client_handlers.requests, "post")
        self.mock_post = patcher.start()
        self.addCleanup(patcher.stop)

    def test_sends_openai_style_request(self):
        self.mock_post.return_value = fake_requests_response(200, json.dumps(OK_RESPONSE))

        response = make_client().invoke("hi", temperature=0.2)

        self.assertEqual(response, OK_RESPONSE)
        kwargs = self.mock_post.call_args.kwargs
        self.assertEqual(kwargs["url"], BASE_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "m")
        self.assertEqual(kwargs["json"]["temperature"], 0.2)
        self.assertEqual(kwargs["json"]["messages"][-1], {"role": "user", "content": "hi"})
        self.assertEqual(kwargs["timeout"], llm_client_handlers.REQUEST_TIMEOUT_SECONDS)

    def test_error_status_shows_status_and_provider_message(self):
        self.mock_post.return_value = fake_requests_response(401, '{"error": "Invalid API key"}')

        with self.assertRaisesRegex(LlmClientError, "HTTP 401.*Invalid API key"):
            make_client().invoke("hi")

    def test_connection_error(self):
        self.mock_post.side_effect = requests.ConnectionError("connection refused")

        with self.assertRaisesRegex(LlmClientError, "Could not reach the LLM"):
            make_client().invoke("hi")

    def test_timeout(self):
        self.mock_post.side_effect = requests.Timeout("read timed out")

        with self.assertRaisesRegex(LlmClientError, "Could not reach the LLM"):
            make_client().invoke("hi")

    def test_invalid_json(self):
        self.mock_post.return_value = fake_requests_response(200, "<html>")

        with self.assertRaisesRegex(LlmClientError, "invalid JSON"):
            make_client().invoke("hi")

    def test_invalid_prompt_is_not_sent(self):
        with self.assertRaises(ValueError):
            make_client().invoke([{"role": "assistant", "content": "no user message"}])

        self.mock_post.assert_not_called()


class TestStream(unittest.TestCase):
    """
    Tests for the sync ``stream`` method.
    """

    def setUp(self):
        patcher = patch.object(llm_client_handlers.requests, "post")
        self.mock_post = patcher.start()
        self.addCleanup(patcher.stop)

    def test_parses_sse_into_json_chunks(self):
        self.mock_post.return_value = fake_requests_response(
            200, b"".join(SSE_CHUNKS).decode()
        )

        chunks = list(make_client().stream("hi"))

        self.assertEqual(chunks, SSE_CHUNKS_EXPECTED)
        kwargs = self.mock_post.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["stream_options"], {"include_usage": True})

    def test_caller_can_override_stream_options(self):
        self.mock_post.return_value = fake_requests_response(200, "data: [DONE]\n\n")

        list(make_client().stream("hi", stream_options={"include_usage": False}))

        self.assertEqual(
            self.mock_post.call_args.kwargs["json"]["stream_options"],
            {"include_usage": False},
        )

    def test_stops_at_done(self):
        self.mock_post.return_value = fake_requests_response(
            200, 'data: {"n": 1}\n\ndata: [DONE]\n\ndata: not json\n\n'
        )

        self.assertEqual(list(make_client().stream("hi")), [{"n": 1}])

    def test_invalid_json_event(self):
        self.mock_post.return_value = fake_requests_response(200, "data: oops\n\n")

        with self.assertRaisesRegex(LlmClientError, "streamed invalid JSON"):
            list(make_client().stream("hi"))

    def test_error_status(self):
        self.mock_post.return_value = fake_requests_response(500, "boom")

        with self.assertRaisesRegex(LlmClientError, "HTTP 500.*boom"):
            list(make_client().stream("hi"))


class TestAinvoke(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the async ``ainvoke`` method.
    """

    async def test_sends_openai_style_request(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = json.loads(request.read())
            return httpx.Response(200, json=OK_RESPONSE)

        with patch_async_server(handler):
            response = await make_client().ainvoke("hi", temperature=0.2)

        self.assertEqual(response, OK_RESPONSE)
        self.assertEqual(seen["auth"], "Bearer test-key")
        self.assertEqual(seen["body"]["model"], "m")
        self.assertEqual(seen["body"]["temperature"], 0.2)

    async def test_error_status_shows_status_and_provider_message(self):
        handler = lambda request: httpx.Response(401, json={"error": "Invalid API key"})

        with patch_async_server(handler):
            with self.assertRaisesRegex(LlmClientError, "HTTP 401.*Invalid API key"):
                await make_client().ainvoke("hi")

    async def test_connection_error(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        with patch_async_server(handler):
            with self.assertRaisesRegex(LlmClientError, "Could not reach the LLM"):
                await make_client().ainvoke("hi")

    async def test_invalid_json(self):
        handler = lambda request: httpx.Response(200, text="<html>")

        with patch_async_server(handler):
            with self.assertRaisesRegex(LlmClientError, "invalid JSON"):
                await make_client().ainvoke("hi")


class TestAstream(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the async ``astream`` method.
    """

    async def collect(self, handler):
        with patch_async_server(handler):
            return [chunk async for chunk in make_client().astream("hi")]

    async def test_parses_chunks_split_anywhere(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.read())
            return httpx.Response(200, content=async_chunks(SSE_CHUNKS))

        self.assertEqual(await self.collect(handler), SSE_CHUNKS_EXPECTED)
        self.assertTrue(seen["body"]["stream"])
        self.assertEqual(seen["body"]["stream_options"], {"include_usage": True})

    async def test_same_result_one_byte_at_a_time(self):
        raw = b"".join(SSE_CHUNKS)
        one_byte_chunks = [raw[i : i + 1] for i in range(len(raw))]
        handler = lambda request: httpx.Response(
            200, content=async_chunks(one_byte_chunks)
        )

        self.assertEqual(await self.collect(handler), SSE_CHUNKS_EXPECTED)

    async def test_stops_at_done(self):
        handler = lambda request: httpx.Response(
            200,
            content=async_chunks(
                [b'data: {"n": 1}\n\ndata: [DONE]\n\n', b"data: not json\n\n"]
            ),
        )

        self.assertEqual(await self.collect(handler), [{"n": 1}])

    async def test_invalid_json_event(self):
        handler = lambda request: httpx.Response(200, text="data: [1, 2\n\n")

        with self.assertRaisesRegex(LlmClientError, "streamed invalid JSON"):
            await self.collect(handler)

    async def test_non_object_event(self):
        handler = lambda request: httpx.Response(200, text="data: [1, 2]\n\n")

        with self.assertRaisesRegex(LlmClientError, "unexpected chunk"):
            await self.collect(handler)

    async def test_error_status(self):
        handler = lambda request: httpx.Response(500, text="boom")

        with self.assertRaisesRegex(LlmClientError, "HTTP 500.*boom"):
            await self.collect(handler)

    async def test_connection_error(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        with self.assertRaisesRegex(LlmClientError, "Could not reach the LLM"):
            await self.collect(handler)


class TestContextWindowCheck(unittest.TestCase):
    """
    Tests for the context window check that runs before any request.
    """

    def setUp(self):
        patcher = patch.object(llm_client_handlers.requests, "post")
        self.mock_post = patcher.start()
        self.addCleanup(patcher.stop)

    def make_small_client(self):
        return HTTPLlmClientHandler(
            base_url=BASE_URL,
            api_key="k",
            model="small-model",
            context_window=1000,
            reserved_output_tokens=200,
        )

    def test_500000_words_are_refused_before_sending(self):
        client = HTTPLlmClientHandler(
            base_url=BASE_URL,
            api_key="k",
            model="m",
            context_window=131072,
            reserved_output_tokens=4096,
        )

        with self.assertRaisesRegex(ContextWindowExceededError, "Nothing was sent"):
            client.invoke("word " * 500_000)

        self.mock_post.assert_not_called()

    def test_message_explains_the_numbers(self):
        with self.assertRaisesRegex(
            ContextWindowExceededError,
            "small-model has room for only 800 input tokens "
            r"\(context window 1,000, 200 kept for the answer\)",
        ):
            self.make_small_client().invoke("word " * 2000)

    def test_request_that_fits_is_sent(self):
        self.mock_post.return_value = fake_requests_response(200, json.dumps(OK_RESPONSE))

        self.make_small_client().invoke("hi")

        self.mock_post.assert_called_once()

    def test_every_call_method_checks_first(self):
        client = self.make_small_client()
        too_big = "word " * 2000

        async def run_async_methods():
            with self.assertRaises(ContextWindowExceededError):
                await client.ainvoke(too_big)
            with self.assertRaises(ContextWindowExceededError):
                async for _ in client.astream(too_big):
                    pass

        with self.assertRaises(ContextWindowExceededError):
            list(client.stream(too_big))
        asyncio.run(run_async_methods())
        self.mock_post.assert_not_called()

    def test_no_context_window_skips_the_check(self):
        self.mock_post.return_value = fake_requests_response(200, json.dumps(OK_RESPONSE))

        make_client().invoke("word " * 2000)

        self.mock_post.assert_called_once()


class TestHeaders(unittest.TestCase):
    """
    Tests for the request headers.
    """

    def test_api_key_is_sent_as_bearer_token(self):
        self.assertEqual(make_client().headers, {"Authorization": "Bearer test-key"})

    def test_no_api_key_sends_no_authorization(self):
        client = HTTPLlmClientHandler(base_url=BASE_URL, api_key=None, model="m")

        self.assertEqual(client.headers, {})
