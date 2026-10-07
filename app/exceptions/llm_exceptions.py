class LlmClientError(Exception):
    """
    Raised when a call to the LLM fails.

    Covers connection problems, timeouts, error status codes and responses
    that are not valid JSON or do not have the expected format.
    """


class ContextWindowExceededError(Exception):
    """
    Raised before sending a request that would not fit the model's context window.

    Nothing is sent to the provider when this is raised.
    """
