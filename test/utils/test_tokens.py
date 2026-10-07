import unittest

from app.utils.tokens import (
    TOKENS_PER_MESSAGE,
    TOKENS_PER_REPLY,
    count_message_tokens,
    count_single_message_tokens,
    count_text_tokens,
    get_encoding,
)


class TestCountTextTokens(unittest.TestCase):
    """
    Tests for ``count_text_tokens``.
    """

    def test_empty_text_has_no_tokens(self):
        self.assertEqual(count_text_tokens(""), 0)

    def test_short_word_is_one_token(self):
        self.assertEqual(count_text_tokens("hello"), 1)

    def test_roughly_four_characters_per_token(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20

        tokens = count_text_tokens(text)

        self.assertLess(abs(len(text) / tokens - 4), 1.5)

    def test_special_token_text_is_counted_as_plain_text(self):
        # Must not raise for text that looks like a special token.
        self.assertGreater(count_text_tokens("<|endoftext|>"), 0)

    def test_encoding_is_loaded_once(self):
        self.assertIs(get_encoding(), get_encoding())


class TestCountMessageTokens(unittest.TestCase):
    """
    Tests for ``count_single_message_tokens`` and ``count_message_tokens``.
    """

    def test_single_message_adds_framing_overhead(self):
        message = {"role": "user", "content": "hello"}

        self.assertEqual(
            count_single_message_tokens(message),
            TOKENS_PER_MESSAGE + count_text_tokens("user") + count_text_tokens("hello"),
        )

    def test_messages_add_reply_overhead(self):
        messages = [
            {"role": "system", "content": "Be short."},
            {"role": "user", "content": "hello"},
        ]

        self.assertEqual(
            count_message_tokens(messages),
            TOKENS_PER_REPLY + sum(count_single_message_tokens(m) for m in messages),
        )

    def test_no_messages(self):
        self.assertEqual(count_message_tokens([]), TOKENS_PER_REPLY)
