import codecs
import re
from dataclasses import dataclass

# A line ends with \r\n, \n or \r
LINE_END = re.compile(r"\r\n|\r|\n")

DONE_MARKER = "[DONE]"


@dataclass(frozen=True)
class SSEEvent:
    """
    One Server-Sent Event.

    Attributes
    ----------
    data : str
        The event data. Several ``data:`` lines are joined with ``\\n``.
    event : str
        The event type from the ``event:`` field, ``"message"`` by default.
    """

    data: str
    event: str = "message"


class SSEParser:
    """
    Turn raw bytes of a Server-Sent Events stream into events.

    Network chunks don't line up with lines or events, so the parser keeps
    whatever is unfinished and waits for the next chunk. It follows the SSE
    format:

    - lines end with ``\\r\\n``, ``\\n`` or ``\\r``
    - a blank line ends an event
    - lines starting with ``:`` are comments and are ignored
    - ``field: value`` lines set a field, and only ``data`` and ``event``
      are used; other fields such as ``id`` and ``retry`` are ignored
    - ``data: [DONE]`` (the OpenAI convention) ends the stream

    Attributes
    ----------
    done : bool
        ``True`` once ``data: [DONE]`` was seen. Later bytes are ignored.

    Examples
    --------
    >>> parser = SSEParser()
    >>> parser.feed(b'data: {"a"')
    []
    >>> parser.feed(b': 1}\\n\\ndata: [DONE]\\n\\n')
    [SSEEvent(data='{"a": 1}', event='message')]
    >>> parser.done
    True
    """

    def __init__(self):
        # Decodes UTF-8 even when a character is split between two chunks
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._data_lines: list[str] = []
        self._event_type = ""
        self.done = False

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        """
        Add a chunk of bytes and return the events it completed.

        Parameters
        ----------
        chunk : bytes
            Raw bytes from the network, split anywhere.

        Returns
        -------
        list[SSEEvent]
            Events that ended in this chunk, in order. Empty if no event
            ended yet, or if the stream is already done.
        """
        if self.done:
            return []

        self._buffer += self._decoder.decode(chunk)
        events = []
        while not self.done:
            line = self._next_line()
            if line is None:
                break
            event = self._process_line(line)
            if event is not None:
                events.append(event)
        return events

    def finish(self) -> list[SSEEvent]:
        """
        Handle the end of the stream.

        The SSE format says an event without its closing blank line should
        be dropped. Some servers skip that last blank line, so the parser is
        lenient and treats the end of the stream as the end of the event.

        Returns
        -------
        list[SSEEvent]
            The last event, if one was still open.
        """
        if self.done:
            return []

        self._buffer += self._decoder.decode(b"", final=True)
        events = self.feed(b"")
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._process_line(line)

        event = self._process_line("")
        if event is not None:
            events.append(event)
        return events

    def _next_line(self) -> str | None:
        """
        Take the next complete line out of the buffer.

        Returns
        -------
        str | None
            The line without its line ending, or ``None`` if the buffer has
            no complete line yet.
        """
        match = LINE_END.search(self._buffer)
        if match is None:
            return None

        # A \r at the very end may be the first half of \r\n, so wait
        if match.group() == "\r" and match.end() == len(self._buffer):
            return None

        line = self._buffer[: match.start()]
        self._buffer = self._buffer[match.end() :]
        return line

    def _process_line(self, line: str) -> SSEEvent | None:
        """
        Apply one line to the event being built.

        Parameters
        ----------
        line : str
            One line without its line ending.

        Returns
        -------
        SSEEvent | None
            The finished event if this line was the blank line that ends it,
            otherwise ``None``.
        """
        if line == "":
            return self._dispatch()

        if line.startswith(":"):
            return None  # comment, e.g. ": ping" keep-alives

        field, _, value = line.partition(":")
        value = value.removeprefix(" ")  # only one leading space is dropped

        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event_type = value
        return None

    def _dispatch(self) -> SSEEvent | None:
        """
        Finish the event being built.

        Returns
        -------
        SSEEvent | None
            The event, or ``None`` if it had no data or was ``[DONE]``.
        """
        if not self._data_lines:
            self._event_type = ""
            return None

        data = "\n".join(self._data_lines)
        event = SSEEvent(data=data, event=self._event_type or "message")
        self._data_lines = []
        self._event_type = ""

        if data == DONE_MARKER:
            self.done = True
            return None
        return event
