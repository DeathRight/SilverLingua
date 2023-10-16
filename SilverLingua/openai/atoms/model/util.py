from typing import Literal, Optional

from SilverLingua.core.atoms.tool.util import FunctionCall

# List of OpenAI models and their maximum number of tokens.
OpenAIModels = {
    "text-embedding-ada-002": 8191,
    "text-davinci-003": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
}

OpenAIChatModels = {k: v for k, v in OpenAIModels.items() if k.startswith("gpt-")}


class ChatCompletionInputMessage:
    """
    A message sent to the OpenAI chat completion API.

    Attributes:
        role: The role of the message. One of "system",
        "user", "assistant", or "function".
        content: The content of the message. May be None if function_call is not None.
        name: The name of the function to call if role is "function".
        function_call: The name and arguments of a function that should be called,
        as generated by the model.
    """

    role: str
    content: Optional[str]
    name: Optional[str]
    function_call: Optional[FunctionCall]

    def __init__(
        self,
        role: str,
        content: Optional[str] = None,
        name: Optional[str] = None,
        function_call: Optional[FunctionCall] = None,
    ) -> None:
        self.role = role
        self.content = content
        self.name = name
        self.function_call = function_call


class ChatCompletionOutputMessage:
    """
    A message returned by the OpenAI chat completion API.

    Attributes:
        role: The role of the message.
        content: The content of the message.
        function_call: The name and arguments of a function that should be called,
        as generated by the model.
    """

    role: str
    content: Optional[str]
    function_call: Optional[FunctionCall]


class ChatCompletionChoice:
    """
    A choice returned by the OpenAI chat completion API.

    Attributes:
        index: The index of the choice in the list of choices.
        message: A chat completion message generated by the model.
        finish_reason: The reason the model stopped generating tokens. This will be stop
        if the model hit a natural stop point or a provided stop sequence,
        length if the maximum number of tokens specified in the request was reached,
        or function_call if the model called a function.
    """

    index: int
    message: ChatCompletionOutputMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionUsage:
    """
    Usage statistics for the OpenAI chat completion request.

    Attributes:
        prompt_tokens: The number of tokens in the prompt.
        completion_tokens: The number of tokens in the generated completion.
        total_tokens: The total number of tokens used. (prompt + completion)
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionOutput:
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
