import json
import unittest

from app.utils.sse_parser import SSEEvent, SSEParser

# The test data from challenge 1.2
CHALLENGE_CHUNKS = [
    b": ping\n\n",
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
    b'\ndata: {"choices":[{"delta":{"content":"lo"}}]}\n\nda',
    b"ta: [DONE]\n\n",
]


def parse(chunks):
    """
    Feed every chunk, then finish, and return all events and the parser.
    """
    parser = SSEParser()
    events = []
    for chunk in chunks:
        events += parser.feed(chunk)
    events += parser.finish()
    return events, parser


def text_of(events):
    return "".join(
        json.loads(e.data)["choices"][0]["delta"]["content"] for e in events
    )


class TestChallengeData(unittest.TestCase):
    """
    The two proofs from challenge 1.2.
    """

    def test_chunks_give_hello_and_finish(self):
        events, parser = parse(CHALLENGE_CHUNKS)

        self.assertEqual(text_of(events), "Hello")
        self.assertTrue(parser.done)

    def test_one_byte_at_a_time_gives_the_same_result(self):
        raw = b"".join(CHALLENGE_CHUNKS)

        events, parser = parse([raw[i : i + 1] for i in range(len(raw))])

        self.assertEqual(text_of(events), "Hello")
        self.assertTrue(parser.done)


class TestSSEParser(unittest.TestCase):
    """
    Tests for ``SSEParser`` and the SSE format rules.
    """

    def test_event_ends_only_at_blank_line(self):
        parser = SSEParser()

        self.assertEqual(parser.feed(b"data: hi\n"), [])
        self.assertEqual(parser.feed(b"\n"), [SSEEvent(data="hi")])

    def test_comments_are_ignored(self):
        events, _ = parse([b": keep-alive\n: another\ndata: x\n\n"])

        self.assertEqual(events, [SSEEvent(data="x")])

    def test_several_data_lines_are_joined_with_newline(self):
        events, _ = parse([b"data: line 1\ndata: line 2\n\n"])

        self.assertEqual(events, [SSEEvent(data="line 1\nline 2")])

    def test_only_one_leading_space_is_dropped(self):
        events, _ = parse([b"data:no space\n\ndata:  two spaces\n\n"])

        self.assertEqual([e.data for e in events], ["no space", " two spaces"])

    def test_data_without_colon_is_empty_data(self):
        events, _ = parse([b"data\n\n"])

        self.assertEqual(events, [SSEEvent(data="")])

    def test_event_type(self):
        events, _ = parse([b"event: update\ndata: x\n\ndata: y\n\n"])

        self.assertEqual(
            events, [SSEEvent(data="x", event="update"), SSEEvent(data="y")]
        )

    def test_other_fields_are_ignored(self):
        events, _ = parse([b"id: 7\nretry: 1000\nfoo: bar\ndata: x\n\n"])

        self.assertEqual(events, [SSEEvent(data="x")])

    def test_blank_lines_without_data_make_no_event(self):
        events, _ = parse([b"\n\n\nid: 1\n\n"])

        self.assertEqual(events, [])

    def test_all_line_endings(self):
        for name, raw in {
            "LF": b"data: a\n\ndata: b\n\n",
            "CRLF": b"data: a\r\n\r\ndata: b\r\n\r\n",
            "CR": b"data: a\r\rdata: b\r\r",
        }.items():
            with self.subTest(name):
                events, _ = parse([raw])
                self.assertEqual([e.data for e in events], ["a", "b"])

    def test_crlf_split_between_chunks(self):
        events, _ = parse([b"data: a\r", b"\n\r", b"\ndata: b\r\n\r\n"])

        self.assertEqual([e.data for e in events], ["a", "b"])

    def test_utf8_character_split_between_chunks(self):
        raw = "data: héllo 👋\n\n".encode()
        split_at = raw.index("👋".encode()) + 2  # in the middle of the emoji

        events, _ = parse([raw[:split_at], raw[split_at:]])

        self.assertEqual(events, [SSEEvent(data="héllo 👋")])

    def test_nothing_after_done(self):
        parser = SSEParser()

        events = parser.feed(b"data: a\n\ndata: [DONE]\n\ndata: b\n\n")

        self.assertEqual(events, [SSEEvent(data="a")])
        self.assertTrue(parser.done)
        self.assertEqual(parser.feed(b"data: c\n\n"), [])
        self.assertEqual(parser.finish(), [])

    def test_done_must_be_the_whole_data(self):
        events, parser = parse([b"data: [DONE] soon\n\n"])

        self.assertEqual(events, [SSEEvent(data="[DONE] soon")])
        self.assertFalse(parser.done)

    def test_finish_keeps_last_event_without_blank_line(self):
        events, _ = parse([b"data: a\n\ndata: b"])

        self.assertEqual([e.data for e in events], ["a", "b"])

    def test_finish_sees_done_without_blank_line(self):
        _, parser = parse([b"data: [DONE]\n"])

        self.assertTrue(parser.done)

    def test_not_done_without_done_marker(self):
        _, parser = parse([b"data: a\n\n"])

        self.assertFalse(parser.done)
