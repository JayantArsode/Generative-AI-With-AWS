import argparse
import asyncio
import sys
from app.agents.agent_with_history import call_agent_with_history
from app.config import get_provider
from app.exceptions import (
    ContextWindowExceededError,
    LlmClientError,
    ProviderConfigError,
)
from app.utils.formatting import format_answer_stats
from app.schemas.provider_config import ProviderConfig

# Errors we expect and can explain in one line. Anything else is a bug.
USER_FACING_ERRORS = (
    LlmClientError,
    ContextWindowExceededError,
    ProviderConfigError,
    ValueError,
)


def print_error(error: Exception) -> None:
    """
    Print an expected error as one readable line on stderr.

    Parameters
    ----------
    error : Exception
        The error to print.
    """
    print(f"Error: {error}", file=sys.stderr)


async def ask(provider: ProviderConfig, question: str) -> int:
    """
    Ask one question without any chat history and print the answer.

    Parameters
    ----------
    provider : ProviderConfig
        The provider and model to call.
    question : str
        The question to ask.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    try:
        answer = await call_agent_with_history(
            provider=provider, user_query=question, use_history=False
        )
    except USER_FACING_ERRORS as e:
        print_error(e)
        return 1

    print(answer.text, flush=True)
    print(format_answer_stats(answer), file=sys.stderr)
    return 0


async def chat(provider: ProviderConfig, use_history: bool = True) -> int:
    """
    Chat with the LLM from the console.

    Each prompt is read from the console and the LLM reply is printed with
    its stats. Errors are printed so the chat can continue. The chat ends on
    Ctrl+C or Ctrl+Z / Ctrl+D (end of input).

    Parameters
    ----------
    provider : ProviderConfig
        The provider and model to call.
    use_history : bool, optional
        Send the earlier turns with each question, by default True.

    Returns
    -------
    int
        Exit code, always 0.
    """
    history_note = "history on" if use_history else "history off"
    print(
        f"Chatting with {provider.name} ({provider.model}), {history_note}. "
        "Ctrl+C to quit.",
        file=sys.stderr,
    )

    while True:
        try:
            user_query = input("$ ").strip()
        except EOFError:
            return 0

        if not user_query:
            continue

        try:
            answer = await call_agent_with_history(
                provider=provider, user_query=user_query, use_history=use_history
            )
        except USER_FACING_ERRORS as e:
            print_error(e)
            continue

        print(">", answer.text, flush=True)
        print(format_answer_stats(answer), file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser for ``cirrus``.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``ask`` and ``chat`` commands.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--provider",
        help="provider name from providers.yml (default: the file's default)",
    )

    parser = argparse.ArgumentParser(
        prog="cirrus", description="Talk to any OpenAI-compatible model."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask_parser = commands.add_parser(
        "ask", parents=[shared], help="ask one question, with no history"
    )
    ask_parser.add_argument(
        "question", help="the question to ask, or - to read it from stdin"
    )

    chat_parser = commands.add_parser(
        "chat", parents=[shared], help="chat in the console, keeping the history"
    )
    chat_parser.add_argument(
        "--no-history",
        action="store_true",
        help="send only the latest question, so earlier turns are forgotten",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    """
    Entry point of the ``cirrus`` command.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line arguments. ``None`` reads them from ``sys.argv``.

    Returns
    -------
    int
        Exit code.
    """
    args = build_parser().parse_args(argv)

    try:
        provider = get_provider(args.provider)
    except ProviderConfigError as e:
        print_error(e)
        return 1

    try:
        if args.command == "ask":
            question = sys.stdin.read() if args.question == "-" else args.question
            return asyncio.run(ask(provider, question))
        return asyncio.run(chat(provider, use_history=not args.no_history))
    except KeyboardInterrupt:
        print("\nBye!", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(run())
