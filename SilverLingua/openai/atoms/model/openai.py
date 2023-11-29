import json
import logging
import os
from typing import Any, Callable, Coroutine, List, Optional, Union

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
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParams,
    CompletionCreateParamsNonStreaming,
)
from pydantic import ConfigDict, Field

from SilverLingua.core.atoms.memory import Idearium, Notion
from SilverLingua.core.atoms.memory.idearium import Tokenizer
from SilverLingua.core.atoms.model import Model, ModelType
from SilverLingua.core.atoms.role import ChatRole

from ..role import OpenAIChatRole
from .util import CompletionParams, OpenAIModelName, OpenAIModels

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """
    An OpenAI model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # init parameters
    name: OpenAIModelName
    # Completion parameters
    completion_params: CompletionParams = Field(
        default_factory=CompletionParams,
        description="Parameters used when calling the OpenAI completions API.",
    )

    # OpenAI clients
    client: OpenAI
    client_async: AsyncOpenAI
    # Model parameters
    role: OpenAIChatRole = OpenAIChatRole
    llm: Completions | Embeddings
    llm_async: AsyncCompletions | AsyncEmbeddings

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
    def max_tokens(self) -> int:
        # Subtract the max response from the maximum number of tokens
        # to leave room for the response.
        return OpenAIModels[self.name] - (self.completion_params.max_tokens or 124)

    @property
    def __chat_args(self):
        """
        Boilerplate arguments for the OpenAI chat completion API to be unpacked.
        """
        return {
            **self.completion_params.model_dump(exclude_none=True),
            "model": self.name,
        }

    def _preprocess(self, messages: List[Notion]):
        msgs: List[Notion] = []
        for msg in messages:
            msgs.append(
                Notion(
                    content=msg.content,
                    role=str(OpenAIChatRole[msg.chat_role.name].value),
                    persistent=msg.persistent,
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
                    if msg.content != "":
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

                        # Remove any tool calls that don't have a string ID
                        tool_calls = [
                            tool_call
                            for tool_call in tool_calls
                            if isinstance(tool_call["id"], str)
                        ]

                        # If there are no tool calls left, skip this message
                        if len(tool_calls) == 0:
                            continue

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
                                content=json.dumps(
                                    msg.model_dump(include="tool_calls")["tool_calls"]
                                ),
                                role=str(ChatRole.TOOL_CALL.value),
                            )
                        )
                    else:
                        if msg.content is not None:
                            output.append(
                                Notion(content=msg.content, role=str(ChatRole.AI.value))
                            )
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
                                content=json.dumps(
                                    msg.model_dump(include="tool_calls")["tool_calls"]
                                ),
                                role=str(ChatRole.TOOL_CALL.value),
                            )
                        )
                    else:
                        if msg.content is not None:
                            output.append(
                                Notion(content=msg.content, role=str(ChatRole.AI.value))
                            )
            else:
                raise ValueError(
                    "Invalid response - has neither message nor delta"
                    + "property set in choices."
                    + "\n response.choices[0]: "
                    + "{response.choices[0].model_dump_json(exclude_none=True)}"
                )
        elif self.type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding models are not yet supported.")
        elif self.type == ModelType.CODE:
            raise NotImplementedError("Code models are not yet supported.")
        return output

    def _postprocess(self, response: List[Notion]) -> List[Notion]:
        return response

    def _retry_call(
        self,
        input: List[ChatCompletionMessageParam],
        e: Exception,
        api_call: Callable,
        retries: int = 0,
    ):
        if input is None:
            raise ValueError("No input provided.")

        if not isinstance(input, list):
            raise ValueError("Input must be a list of ChatCompletionMessageParam.")

        inp = input.copy()
        # Remove everything since the last user message
        for i in range(len(inp) - 1, -1, -1):
            if inp[i]["role"] == str(OpenAIChatRole.HUMAN.value):
                inp = inp[: i + 1]
                break
        # Inform the AI
        inp.append(
            {
                "role": str(OpenAIChatRole.SYSTEM.value),
                "content": f"Error calling OpenAI chat completion API: {e}. "
                + "Do not try to repeat the last action.",
            }
        )
        # Try again
        retries += 1
        return self._common_call_logic(inp, api_call, retries)

    def _call(
        self,
        input: List[ChatCompletionMessageParam],
        retries: int = 0,
        **kwargs,
    ) -> ChatCompletion:
        def api_call(**kwargs_):
            return self.llm.create(**kwargs, **kwargs_)

        return self._common_call_logic(input, api_call, retries)

    async def _acall(
        self,
        input: List[ChatCompletionMessageParam],
        retries: int = 0,
        **kwargs,
    ) -> Coroutine[Any, Any, ChatCompletion]:
        async def api_call(**kwargs_):
            return await self.llm_async.create(**kwargs, **kwargs_)

        return await self._common_call_logic(input, api_call, retries)

    def generate(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParamsNonStreaming = None,
    ):
        create_params = create_params or {}
        return self._common_generate_logic(messages, False, create_params=create_params)

    async def agenerate(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParamsNonStreaming = None,
    ):
        create_params = create_params or {}
        return await self._common_generate_logic(
            messages, True, create_params=create_params
        )

    def stream(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParams = None,
    ):
        create_params = create_params or {}

        input = self._common_stream_logic(messages)
        output_stream: Stream[ChatCompletionChunk] = self._call(
            input, {**create_params, "stream": True}
        )

        for chunk in output_stream:
            standardized_response = self._postprocess(self._standardize_response(chunk))
            for notion in standardized_response:
                yield notion

    async def astream(
        self,
        messages: Union[Idearium, List[Notion]],
        create_params: CompletionCreateParams = None,
    ):
        create_params = create_params or {}

        input = self._common_stream_logic(messages)
        output_stream: AsyncStream[ChatCompletionChunk] = await self._acall(
            input, {**create_params, "stream": True}
        )

        async for chunk in output_stream:
            standardized_response = self._postprocess(self._standardize_response(chunk))
            for notion in standardized_response:
                yield notion

    def __init__(
        self,
        name: OpenAIModelName,
        api_key: Optional[str] = None,
        completion_params: Optional[CompletionParams] = None,
    ):
        """
        Creates a new OpenAI model.

        Args:
            name (str): The name of the OpenAI model to use.
            api_key (str, optional): The OpenAI API key to use.
                If None, the `OPENAI_API_KEY` environment variable will be used.
                If that is not set, an exception will be raised.
            completion_params (CompletionParams, optional): Parameters used when calling
                the OpenAI completions API.
                If None, default values will be used.
        """
        completion_params = completion_params or CompletionParams()
        tokenizer = tiktoken.encoding_for_model(name)
        args = {
            "name": name,
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
            "completion_params": completion_params,
            "client": None,
            "client_async": None,
            "llm": None,
            "llm_async": None,
            "can_stream": None,
            "type": None,
            "tokenizer": Tokenizer(encode=tokenizer.encode, decode=tokenizer.decode),
        }

        if args["api_key"] is None:
            raise ValueError(
                "No OpenAI API key provided and "
                + "`OPENAI_API_KEY` not set as an environment variable."
            )

        if completion_params.max_tokens is not None:
            args["max_response"] = completion_params.max_tokens

        args["client"] = OpenAI(api_key=args["api_key"])
        args["client_async"] = AsyncOpenAI(api_key=args["api_key"])

        if args["name"] == "text-embedding-ada-002":
            args["can_stream"] = False
            args["llm"] = args["client"].embeddings
            args["llm_async"] = args["client_async"].embeddings
            args["type"] = ModelType.EMBEDDING
        elif args["name"].lower().find("gpt") != -1:
            args["can_stream"] = True
            args["llm"] = args["client"].embeddings
            args["llm_async"] = args["client_async"].embeddings
            args["type"] = ModelType.CHAT
        else:
            raise ValueError(
                f"Invalid OpenAI model name: {args['name']}.\n"
                + "Only chat and embedding models are supported."
            )
        super().__init__(**args)
