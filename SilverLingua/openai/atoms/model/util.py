import json
from typing import List, Literal, Optional

from SilverLingua.core.atoms.tool.util import FunctionCall, FunctionJSONSchema

# List of OpenAI models and their maximum number of tokens.
OpenAIModels = {
    "text-embedding-ada-002": 8191,
    "text-davinci-003": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-1106-preview": 128000,
}

OpenAIChatModels = {k: v for k, v in OpenAIModels.items() if k.startswith("gpt-")}


class ToolChoiceFunction:
    name: str


class ChatCompletionToolChoice:
    type: Literal["function"] = "function"
    function: ToolChoiceFunction

    def __init__(self, function: ToolChoiceFunction) -> None:
        self.function = function

    def to_dict(self):
        return {
            "type": self.type,
            "function": {"name": self.function.name},
        }


class ChatCompletionInputTool:
    type: Literal["function"] = "function"
    function: FunctionJSONSchema

    def __init__(self, function: FunctionJSONSchema) -> None:
        self.function = function

    def to_dict(self):
        return {
            "type": self.type,
            "function": self.function,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


####################


class ChatCompletionMessageTool:
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

    def __init__(self, id: str, function: FunctionCall) -> None:
        self.id = id
        self.function = function

    def to_json(self):
        return json.dumps(
            {
                "id": self.id,
                "type": self.type,
                "function": self.function.to_dict(),
            }
        )

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict(),
        }


class ChatCompletionMessageToolCalls(List):
    _tool_calls: List[ChatCompletionMessageTool] = []

    def __init__(self, tool_calls: List[ChatCompletionMessageTool]) -> None:
        self._tool_calls = tool_calls

    def to_json(self):
        return json.dumps(self._tool_calls)

    @classmethod
    def from_json(cls, json_str: str):
        return cls(json.loads(json_str))


class ChatCompletionInputMessageToolResponse:
    """
    A response to a tool call in a message sent to the OpenAI chat completion API.

    Attributes:
        tool_call_id: The ID of the tool call the message is a response to.
        name: The name of the tool the message is a response to.
        content: The return of the tool call.
    """

    tool_call_id: str
    name: str
    content: str

    def __init__(self, tool_call_id: str, name: str, content: str) -> None:
        self.tool_call_id = tool_call_id
        self.name = name
        self.content = content

    def to_json(self):
        return json.dumps(
            {
                "tool_call_id": self.tool_call_id,
                "name": self.name,
                "content": self.content,
            }
        )

    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))


class ChatCompletionInputMessage:
    """
    A message sent to the OpenAI chat completion API.

    Attributes:
        role: The role of the message. One of "system",
        "user", "assistant", or "function".
        content: The content of the message. May be None if tool_calls is not None.
        tool_calls: The tools (functions) the model is trying to call.
        (Not used when role is "tool")
        tool_call_id: The ID of the tool call the message is a response to.
        (Used only when role is "tool")
        name: The name of the tool the message is a response to.
        (Used only when role is "tool")
    """

    role: str
    content: Optional[str]
    tool_calls: Optional[List[ChatCompletionMessageTool]]

    def __init__(
        self,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[ChatCompletionMessageTool]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.name = name

    def to_dict(self):
        _dict = {"role": self.role}

        if self.content:
            _dict["content"] = self.content
        if self.tool_calls:
            if isinstance(self.tool_calls[0], dict):
                _dict["tool_calls"] = self.tool_calls
            else:
                _dict["tool_calls"] = [
                    tool_call.to_dict() for tool_call in self.tool_calls
                ]
        if self.tool_call_id:
            _dict["tool_call_id"] = self.tool_call_id
        if self.name:
            _dict["name"] = self.name
        return _dict

    def to_json(self):
        return json.dumps(self.to_dict())


class ChatCompletionOutputMessage:
    """
    A message returned by the OpenAI chat completion API.

    Attributes:
        role: The role of the message.
        content: The content of the message.
        tool_calls: The tools (functions) the model is trying to call.
    """

    role: str
    content: Optional[str]
    tool_calls: Optional[List[ChatCompletionMessageTool]]


####################


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
