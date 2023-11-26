import json
import logging
import os
from typing import Any, Coroutine, List, Optional, Union

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
from pydantic import Field

from SilverLingua.core.atoms.memory import Idearium, Notion
from SilverLingua.core.atoms.model import Model, ModelType
from SilverLingua.core.atoms.role import ChatRole

from ..role import OpenAIChatRole
from .util import CompletionParams, OpenAIModelName, OpenAIModels

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """
    An OpenAI model.
    """

    # init parameters
    name: OpenAIModelName
    # Completion parameters
    completion_params: CompletionParams = Field(
        default_factory=CompletionParams,
        description="Parameters used when calling the OpenAI completions API.",
    )

    # OpenAI clients
    _client: OpenAI
    _client_async: AsyncOpenAI
    # Model parameters
    _role = OpenAIChatRole
    _model: Completions | Embeddings
    _model_async: AsyncCompletions | AsyncEmbeddings

    @property
    def moderation(self) -> Moderations:
        """
        The moderation object used to check if text violates OpenAI's content policy.
        """
        return self._client.moderations

    @property
    def moderation_async(self) -> AsyncModerations:
        """
        The moderation object used to check if text violates OpenAI's content policy.
        """
        return self._client_async.moderations

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
        return self.completion_params.model_dump()

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
                                json.dumps(
                                    msg.model_dump(include="tool_calls")["tool_calls"]
                                ),
                                str(ChatRole.TOOL_CALL.value),
                            )
                        )
                    else:
                        if msg.content is not None:
                            output.append(Notion(msg.content, str(ChatRole.AI.value)))
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
                        if msg.content is not None:
                            output.append(Notion(msg.content, str(ChatRole.AI.value)))
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

            out: Union[ChatCompletion, Stream[ChatCompletionChunk]]
            try:
                out = self._client.chat.completions.create(
                    **self.__chat_args,
                    **create_params,
                    messages=input,
                )
            except Exception as e:
                logger.error(f"Error calling OpenAI chat completion API: {e}")
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
                out = self._call(inp, create_params)

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
                Coroutine[Any, Any, ChatCompletion], AsyncStream[ChatCompletionChunk]
            ]

            try:
                out: Union[
                    Coroutine[Any, Any, ChatCompletion],
                    Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]],
                ] = self._client_async.chat.completions.create(
                    **self.__chat_args, **create_params, messages=input
                )
            except Exception as e:
                logger.error(f"Error calling OpenAI chat completion API: {e}")
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
                out = self._acall(inp, create_params)

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
        args = {}
        args.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if args.api_key is None:
            raise ValueError(
                "No OpenAI API key provided and "
                + "`OPENAI_API_KEY` not set as an environment variable."
            )
        args.name = name
        args.max_response = completion_params.max_tokens
        args.completion_params = completion_params

        args._client = OpenAI(api_key=args.api_key)
        args._client_async = AsyncOpenAI(api_key=args.api_key)

        if args.name == "text-embedding-ada-002":
            args._can_stream = False
            args._model = args._client.embeddings
            args._model_async = args._client_async.embeddings
            args._type = ModelType.EMBEDDING
        elif args.name.lower().find("gpt") != -1:
            args._can_stream = True
            args._model = args._client.chat.completions
            args._model_async = args._client_async.chat.completions
            args._type = ModelType.CHAT
        else:
            raise ValueError(
                f"Invalid OpenAI model name: {args.name}.\n"
                + "Only chat and embedding models are supported."
            )

        super().__init__(**args)
