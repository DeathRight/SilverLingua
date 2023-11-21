import logging
import os
from typing import List, Optional, Union

import openai
import tiktoken

from SilverLingua.core.atoms.memory import Idearium, Notion
from SilverLingua.core.atoms.model import Model, ModelType
from SilverLingua.core.atoms.role import ChatRole

from ..role import OpenAIChatRole
from .util import (
    ChatCompletionInputMessage,
    ChatCompletionInputMessageToolResponse,
    ChatCompletionInputTool,
    ChatCompletionMessageToolCalls,
    ChatCompletionOutput,
    ChatCompletionToolChoice,
    OpenAIModels,
)

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """
    An OpenAI model.
    """

    # Completion parameters
    temperature: float
    top_p: float
    n: int
    stop: Optional[list]
    presence_penalty: float
    frequency_penalty: float
    logit_bias: Optional[dict]
    tools: Optional[List[ChatCompletionInputTool]]

    # Text completion specific parameters
    suffix: Optional[str]
    logprobs: Optional[int]
    echo: bool
    best_of: int

    @property
    def role(self) -> ChatRole:
        return OpenAIChatRole

    @property
    def api_key(self) -> str:
        """
        The API key for the model.

        Defaults to the OPENAI_API_KEY environment variable.
        """
        return self._api_key

    @property
    def moderation(self) -> openai.Moderation:
        """
        The moderation object used to check if text violates OpenAI's content policy.
        """
        return openai.Moderation

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> ModelType:
        return self._type

    @property
    def model(self) -> openai.ChatCompletion | openai.Completion:
        return self._model

    @property
    def can_stream(self) -> bool:
        return self._can_stream

    @property
    def streaming(self) -> bool:
        return self._streaming

    @property
    def max_tokens(self) -> int:
        # Subtract the max response from the maximum number of tokens
        # to leave room for the response.
        return OpenAIModels[self.name] - (
            self.max_response == 0 and 124 or self.max_response
        )

    @property
    def max_response(self) -> int:
        return self._max_response

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self.name)

    @property
    def __chat_args(self):
        """
        Boilerplate arguments for the OpenAI chat completion API to be unpacked.
        """
        return {
            "model": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.streaming,
            "stop": self.stop,
            "max_tokens": self.max_response,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "tools": self.tools,
        }

    @property
    def __text_args(self):
        """
        Boilerplate arguments for the OpenAI text completion API to be unpacked.
        """
        return {
            "model": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_response,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
        }

    def _preprocess(self, messages: List[Notion]):
        for msg in messages:
            # Ensure all roles are OpenAIChatRole members
            msg.chat_role = OpenAIChatRole[msg.chat_role]
        return messages

    def _format_request(
        self, messages: List[Notion]
    ) -> Union[str, List[ChatCompletionInputMessage]]:
        input = None
        if self.type == ModelType.CHAT:
            input: List[ChatCompletionInputMessage] = []
            for msg in messages:
                if msg.chat_role == ChatRole.TOOL_CALL:
                    """
                    # msg.content is the same as "tool_calls" in this case
                    msg.content = [
                        {
                            "id": "0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {
                                    "location": "New York City",
                                }
                            }
                        }
                    ]
                    """
                    tool_calls = ChatCompletionMessageToolCalls.from_json(msg.content)
                    input.append(
                        ChatCompletionInputMessage(
                            role=OpenAIChatRole.TOOL_CALL,
                            tool_calls=tool_calls,
                        )
                    )
                elif msg.chat_role == ChatRole.TOOL_RESPONSE:
                    """
                    msg.content = {
                        "tool_call_id": "0",
                        "name": "get_weather",
                        "content": {
                            "temperature": "70",
                        }
                    }
                    """
                    tool_response = ChatCompletionInputMessageToolResponse.from_json(
                        msg.content
                    )
                    input.append(
                        ChatCompletionInputMessage(
                            role=OpenAIChatRole.TOOL_RESPONSE,
                            tool_call_id=tool_response.tool_call_id,
                            name=tool_response.name,
                            content=tool_response.content,
                        )
                    )
                else:
                    input.append(
                        ChatCompletionInputMessage(
                            role=msg.chat_role, content=msg.content
                        )
                    )
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")
        return input

    def _standardize_response(
        self, response: Union[str, ChatCompletionOutput]
    ) -> List[Notion]:
        output: List[Notion] = []
        if self.type == ModelType.CHAT:
            r: ChatCompletionOutput = response
            for choice in r.choices:
                msg = choice.message
                if hasattr(msg, "tool_calls") and msg.tool_calls is not None:
                    output.append(
                        Notion(
                            ChatCompletionMessageToolCalls(msg.tool_calls).to_json(),
                            ChatRole.TOOL_CALL,
                        )
                    )
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id is not None:
                    output.append(
                        Notion(
                            ChatCompletionInputMessageToolResponse(
                                msg.tool_call_id, msg.name, msg.content
                            ).to_json(),
                            ChatRole.TOOL_RESPONSE,
                        )
                    )
                else:
                    output.append(Notion(msg.content, ChatRole.AI))
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")
        return output

    def _postprocess(self, response: List[Notion]) -> List[Notion]:
        return response

    def _call(
        self,
        input: Union[str, List[ChatCompletionInputMessage], List[str], List[List[int]]],
        tool_choice: Optional[ChatCompletionToolChoice] = None,
        **kwargs,
    ):
        if input is None:
            raise ValueError("No input provided.")

        if self.type == ModelType.CHAT:
            if input is not list:
                raise ValueError("Input must be a list of ChatCompletionInputMessage.")
            return self.model.create(
                **self.__chat_args, messages=input, tool_choice=tool_choice, **kwargs
            )
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")

    async def _acall(
        self,
        input: Union[str, List[ChatCompletionInputMessage], List[str], List[List[int]]],
        tool_choice: Optional[ChatCompletionToolChoice] = None,
        **kwargs,
    ):
        if input is None:
            raise ValueError("No input provided.")

        if self.type == ModelType.CHAT:
            if input is not list:
                raise ValueError("Input must be a list of ChatCompletionInputMessage.")
            return await self.model.acreate(
                **self.__chat_args, messages=input, tool_choice=tool_choice, **kwargs
            )
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")

    def generate(
        self,
        messages: Union[Idearium, List[Notion]],
        tool_choice: Optional[ChatCompletionToolChoice] = None,
        **kwargs,
    ):
        if messages is None:
            raise ValueError("No messages provided.")

        # If messages is not an Idearium, convert it to one
        # so we can take advantage of its automatic trimming.
        if not isinstance(messages, Idearium):
            messages = Idearium(self.tokenizer, self.max_tokens, messages)

        input = self._format_request(self._preprocess(messages))

        output = self._standardize_response(
            self._call(input, tool_choice=tool_choice, **kwargs)
        )

        return self._postprocess(output)

    def __init__(
        self,
        name: str = "gpt-3.5-turbo",
        tools: List[ChatCompletionInputTool] = None,
        streaming: bool = False,
        max_response: int = 256,
        api_key: str = "",
        top_p: float = 1.0,
        temperature: float = 1.0,
        n: int = 1,
        stop: list = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: dict = None,
        suffix: Optional[str] = None,
        logprobs: Optional[int] = None,
        echo: bool = False,
        best_of: int = 1,
    ):
        """
        Creates an OpenAI model.

        Args:
            name (str, optional): The name of the model version being used.
            Defaults to "gpt-3.5-turbo".
            tools (List[ChatCompletionInputTool], optional): The tools to use.
            Defaults to None.

            streaming (bool, optional): Whether the model should be initialized as
            streaming. Defaults to False.
            max_response (int, optional): The maximum number of tokens the model can
            return. Defaults to 256.
            api_key (str, optional): The API key for the model. Defaults to the
            OPENAI_API_KEY environment variable.
            top_p (float, optional): The nucleus sampling probability. Defaults to 1.0.
            temperature (float, optional): The temperature of the model.
            Defaults to 1.0.
            n (int, optional): The number of completions to generate. Defaults to 1.
            stop (list, optional): The stop sequence(s) for the model. Defaults to None.
            presence_penalty (float, optional): The presence penalty for the model.
            Defaults to 0.0.
            frequency_penalty (float, optional): The frequency penalty for the model.
            Defaults to 0.0.
            logit_bias (dict, optional): The logit bias for the model. Defaults to None.
            suffix (str, optional): The suffix for the model. Defaults to None.
            logprobs (int, optional): The number of logprobs for the model.
            Defaults to None.
            echo (bool, optional): Echo back the prompt in addition to the completion.
            Defaults to False.
            best_of (int, optional): Generates `best_of` completions server-side and
            returns the "best" (the one with the lowest log probability per token).
            Defaults to 1.
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

        if name not in OpenAIModels:
            raise ValueError(f"Invalid OpenAI model name: {name}")
        self._name = name

        if self._name == "text-embedding-ada-002":
            self._can_stream = False
            self._model = openai.Embedding
            self._type = ModelType.EMBEDDING
        elif self._name.lower().find("gpt") != -1:
            self._can_stream = True
            self._model = openai.ChatCompletion
            self._type = ModelType.CHAT
        else:
            raise ValueError(
                f"Invalid OpenAI model name: {name}."
                + "Only chat and embedding models are supported."
            )

        self._streaming = self._can_stream and streaming
        self._max_response = max_response
        if self.max_response == 0:
            self._max_response = self.max_tokens

        # OpenAI specific parameters
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias

        # Text completion specific parameters
        self.suffix = suffix
        self.logprobs = logprobs
        self.echo = echo
        self.best_of = best_of

        # Chat completion specific parameters
        self.tools = tools
