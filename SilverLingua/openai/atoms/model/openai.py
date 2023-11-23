import json
import logging
import os
from typing import Any, Coroutine, List, Optional, Union

import tiktoken
from openai import AsyncOpenAI, OpenAI
from openai._streaming import AsyncStream, Stream
from openai.resources import (
    AsyncCompletions,
    AsyncEmbeddings,
    AsyncModerations,
    Completions,
    Embeddings,
    Moderations,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParams,
    CompletionCreateParamsNonStreaming,
)

from SilverLingua.core.atoms.memory import Idearium, Notion
from SilverLingua.core.atoms.model import Model, ModelType
from SilverLingua.core.atoms.role import ChatRole

from ..role import OpenAIChatRole
from .util import OpenAIModels

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """
    An OpenAI model.
    """

    # OpenAI clients
    client: OpenAI
    client_async: AsyncOpenAI

    # Completion parameters
    temperature: float
    top_p: float
    n: int
    stop: Optional[list]
    presence_penalty: float
    frequency_penalty: float
    logit_bias: Optional[dict]
    tools: Optional[List[ChatCompletionToolParam]]

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
    def moderation(self) -> Moderations:
        """
        The moderation object used to check if text violates OpenAI's content policy.
        """
        return self.client.moderations

    @property
    def moderation_async(self) -> AsyncModerations:
        """
        The moderation object used to check if text violates OpenAI's content policy.
        """
        return self.client_async.moderations

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> ModelType:
        return self._type

    @property
    def model(self) -> Completions | Embeddings:
        return self._model

    @property
    def model_async(self) -> AsyncCompletions | AsyncEmbeddings:
        return self._model_async

    @property
    def can_stream(self) -> bool:
        return self._can_stream

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
        dict = {
            "model": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "max_tokens": self.max_response,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.tools is not None:
            dict["tools"] = self.tools
        if self.stop is not None:
            dict["stop"] = self.stop
        if self.logit_bias is not None:
            dict["logit_bias"] = self.logit_bias
        return dict

    def _preprocess(self, messages: List[Notion]):
        msgs: List[Notion] = []
        for msg in messages:
            msgs.append(
                Notion(
                    msg.content,
                    str(OpenAIChatRole[msg.chat_role.name].value),
                    msg.persistent,
                )
            )
        return msgs

    def _format_request(
        self, messages: List[Notion]
    ) -> Union[str, List[ChatCompletionMessageParam]]:
        input: Union[str, List[ChatCompletionMessageParam]] = []
        if self.type == ModelType.CHAT:
            input: List[ChatCompletionMessageParam] = []
            for msg in messages:
                # logger.debug(f"msg: {msg}")

                msg_content = ""
                try:
                    msg_content = json.loads(msg.content)
                except json.decoder.JSONDecodeError:
                    msg_content = msg.content

                if msg.chat_role == ChatRole.AI:
                    ccim: ChatCompletionAssistantMessageParam
                    if (
                        isinstance(msg_content, list)
                        and len(msg_content) > 0
                        and isinstance(msg_content[0], dict)
                        and "function" in msg_content[0]
                    ):
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
                        tool_calls: List[ChatCompletionMessageToolCall] = msg_content
                        ccim = {
                            "role": str(OpenAIChatRole.TOOL_CALL.value),
                            "tool_calls": tool_calls,
                        }
                    else:
                        ccim = {"role": msg.role, "content": msg.content}
                    input.append(ccim)
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
                    tool_response: ChatCompletionToolMessageParam = {
                        "content": msg_content["content"],
                        "role": str(OpenAIChatRole.TOOL_RESPONSE.value),
                        "tool_call_id": msg_content["tool_call_id"],
                    }
                    input.append(tool_response)
                else:
                    input.append({"role": msg.role, "content": msg.content})
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")
        logger.debug(f"input: {json.dumps(input, indent=2)}")
        return input

    def _standardize_response(
        self, response: Union[str, ChatCompletion, ChatCompletionChunk]
    ) -> List[Notion]:
        output: List[Notion] = []
        if self.type == ModelType.CHAT:
            if (
                hasattr(response.choices[0], "delta")
                and response.choices[0].delta is not None
            ):
                # logger.debug("response is a chunk")
                rc: ChatCompletionChunk = response
                for choice in rc.choices:
                    msg = choice.delta
                    # logger.debug(f"msg: {msg}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls is not None:
                        # logger.debug("msg has tool_calls")
                        output.append(
                            Notion(
                                json.dumps(
                                    msg.model_dump(include="tool_calls")["tool_calls"]
                                ),
                                str(ChatRole.TOOL_CALL.value),
                            )
                        )
                    else:
                        output.append(Notion(msg.content, str(OpenAIChatRole.AI.value)))
            elif (
                hasattr(response.choices[0], "message")
                and response.choices[0].message is not None
            ):
                # logger.debug("response is not a chunk")
                r: ChatCompletion = response
                for choice in r.choices:
                    msg = choice.message
                    # logger.debug(f"msg: {msg}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls is not None:
                        # logger.debug("msg has tool_calls")
                        output.append(
                            Notion(
                                json.dumps(
                                    msg.model_dump(include="tool_calls")["tool_calls"]
                                ),
                                str(ChatRole.TOOL_CALL.value),
                            )
                        )
                    else:
                        output.append(Notion(msg.content, str(OpenAIChatRole.AI.value)))
            else:
                raise ValueError(
                    "Invalid response - has neither message nor delta"
                    + "property set in choices."
                    + f"\n response.choices[0]: {response.choices[0].model_dump_json()}"
                )
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")
        return output

    def _postprocess(self, response: List[Notion]) -> List[Notion]:
        return response

    def _call(
        self,
        input: List[ChatCompletionMessageParam],
        create_params: CompletionCreateParams = None,
    ):
        if create_params is None:
            create_params = {}

        if input is None:
            raise ValueError("No input provided.")

        if self.type == ModelType.CHAT:
            if not isinstance(input, list):
                raise ValueError("Input must be a list of ChatCompletionMessageParam.")

            out: Union[
                ChatCompletion, Stream[ChatCompletionChunk]
            ] = self.client.chat.completions.create(
                **self.__chat_args,
                **create_params,
                messages=input,
            )

            return out
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")

    async def _acall(
        self,
        input: List[ChatCompletionMessageParam],
        create_params: CompletionCreateParams = None,
    ):
        if create_params is None:
            create_params = {}

        if input is None:
            raise ValueError("No input provided.")

        if self.type == ModelType.CHAT:
            if not isinstance(input, list):
                raise ValueError("Input must be a list of ChatCompletionMessageParam.")

            out: Union[
                Coroutine[Any, Any, ChatCompletion],
                Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]],
            ] = self.client_async.chat.completions.create(
                **self.__chat_args, **create_params, messages=input
            )

            return out
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        else:
            raise NotImplementedError("Only chat and embedding models are supported.")

    def generate(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParamsNonStreaming = None,
    ):
        if create_params is None:
            create_params = {}

        if messages is None:
            raise ValueError("No messages provided.")

        # If messages is not an Idearium, convert it to one
        # so we can take advantage of its automatic trimming.
        if not isinstance(messages, Idearium):
            messages = Idearium(self.tokenizer, self.max_tokens, messages)

        input = self._format_request(self._preprocess(messages))

        output = self._standardize_response(self._call(input, create_params))

        return self._postprocess(output)

    async def agenerate(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParamsNonStreaming = None,
    ):
        if create_params is None:
            create_params = {}

        if messages is None:
            raise ValueError("No messages provided.")

        # If messages is not an Idearium, convert it to one
        # so we can take advantage of its automatic trimming.
        if not isinstance(messages, Idearium):
            messages = Idearium(self.tokenizer, self.max_tokens, messages)

        input = self._format_request(self._preprocess(messages))

        output = self._standardize_response(await self._acall(input, create_params))

        return self._postprocess(output)

    def stream(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParams = None,
    ):
        if create_params is None:
            create_params = {}

        if messages is None:
            raise ValueError("No messages provided.")

        # If messages is not an Idearium, convert it to one
        # so we can take advantage of its automatic trimming.
        if not isinstance(messages, Idearium):
            messages = Idearium(self.tokenizer, self.max_tokens, messages)

        input = self._format_request(self._preprocess(messages))

        output_stream: Stream[ChatCompletionChunk] = self._call(
            input, {**create_params, "stream": True}
        )

        for chunk in output_stream:
            standardized_response = self._standardize_response(chunk)

            for notion in standardized_response:
                yield notion

    async def astream(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParams = None,
    ):
        if create_params is None:
            create_params = {}

        if messages is None:
            raise ValueError("No messages provided.")

        # If messages is not an Idearium, convert it to one
        # so we can take advantage of its automatic trimming.
        if not isinstance(messages, Idearium):
            messages = Idearium(self.tokenizer, self.max_tokens, messages)

        input = self._format_request(self._preprocess(messages))

        output_stream: AsyncStream[ChatCompletionChunk] = await self._acall(
            input, {**create_params, "stream": True}
        )

        async for chunk in output_stream:
            standardized_response = self._standardize_response(chunk)

            for notion in standardized_response:
                yield notion

    def __init__(
        self,
        name: str = "gpt-3.5-turbo",
        tools: List[ChatCompletionToolParam] = None,
        max_response: int = 256,
        api_key: str = "",
        top_p: float = 1.0,
        temperature: float = 1.0,
        n: int = 1,
        stop: list = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: dict = None,
    ):
        """
        Creates an OpenAI model.

        Args:
            name (str, optional): The name of the model version being used.
            Defaults to "gpt-3.5-turbo".
            tools (List[ChatCompletionToolParam], optional): The tools to use.
            Defaults to None.
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
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.client_async = AsyncOpenAI(api_key=self.api_key)

        if name is not None and name not in OpenAIModels:
            raise ValueError(f"Invalid OpenAI model name: {name}")
        self._name = name or "gpt-3.5-turbo"

        if self._name == "text-embedding-ada-002":
            self._can_stream = False
            self._model = self.client.embeddings
            self._model_async = self.client_async.embeddings
            self._type = ModelType.EMBEDDING
        elif self._name.lower().find("gpt") != -1:
            self._can_stream = True
            self._model = self.client.chat.completions
            self._model_async = self.client_async.chat.completions
            self._type = ModelType.CHAT
        else:
            raise ValueError(
                f"Invalid OpenAI model name: {name}."
                + "Only chat and embedding models are supported."
            )

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

        # Chat completion specific parameters
        self.tools = tools
