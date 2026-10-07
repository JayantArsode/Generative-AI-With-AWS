from typing import Annotated, Literal
from pydantic import BaseModel, Field, model_validator
from app.prompts.prompt_loaders import get_system_prompt, SystemPrompts


class SystemMessage(BaseModel):
    """
    Message that sets the behaviour of the assistant.

    Attributes
    ----------
    role : Literal["system"]
        Role of the message, always ``"system"``.
    content : str
        Text content of the message.
    """

    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    """
    Message sent by the user.

    Attributes
    ----------
    role : Literal["user"]
        Role of the message, always ``"user"``.
    content : str
        Text content of the message.
    """

    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    """
    Message previously returned by the assistant.

    Attributes
    ----------
    role : Literal["assistant"]
        Role of the message, always ``"assistant"``.
    content : str
        Text content of the message.
    """

    role: Literal["assistant"] = "assistant"
    content: str


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage,
    Field(discriminator="role"),
]


class Messages(BaseModel):
    """
    Ordered conversation sent to the LLM.

    Attributes
    ----------
    messages : list[Message]
        System, user and assistant messages in conversation order.
    """

    messages: list[Message]

    @model_validator(mode="after")
    def check_messages(self) -> "Messages":
        """
        Validate the conversation and add the default system prompt if needed.

        Returns
        -------
        Messages
            The validated conversation.

        Raises
        ------
        ValueError
            If the system message is not first, there is no user message,
            or the last message is not a user message.
        """
        system_count = sum(isinstance(m, SystemMessage) for m in self.messages)
        if system_count == 0:
            self.messages.insert(
                0,
                SystemMessage(content=get_system_prompt(SystemPrompts.DEFAULT_PROMPT)),
            )
        if not isinstance(self.messages[0], SystemMessage):
            raise ValueError("system message must be the first message")
        if not any(isinstance(m, UserMessage) for m in self.messages):
            raise ValueError("at least one user message is required")
        if not isinstance(self.messages[-1], UserMessage):
            raise ValueError("last message must be a user message")
        return self
