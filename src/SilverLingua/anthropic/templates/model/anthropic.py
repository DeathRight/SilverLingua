import json
import logging
import os
from typing import Callable, List, Optional, Type, Union

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageStreamEvent
from pydantic import ConfigDict, Field

from SilverLingua.core.atoms import ChatRole, Tokenizer
from SilverLingua.core.molecules import Notion
from SilverLingua.core.templates.model import Messages, Model, ModelType

from ...atoms import AnthropicChatRole
from .util import AnthropicModelName, AnthropicModels, CompletionParams

logger = logging.getLogger(__name__)


class AnthropicModel(Model):
    """
    An Anthropic model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # init parameters
    name: AnthropicModelName
    # Completion parameters
    completion_params: CompletionParams = Field(
        default_factory=CompletionParams,
        description="Parameters used when calling the Anthropic API.",
    )

    # Anthropic clients
    client: Anthropic
    client_async: AsyncAnthropic
    # Model parameters
    role: Type[ChatRole] = AnthropicChatRole

    @property
    def max_tokens(self) -> int:
        # Get the model's max tokens from AnthropicModels
        return AnthropicModels[self.name]["max_tokens"] - (
            self.completion_params.max_tokens or 1024
        )

    @property
    def __chat_args(self):
        """
        Boilerplate arguments for the Anthropic API to be unpacked.
        """
        args = {
            "model": self.name,
            "max_tokens": self.completion_params.max_tokens,
            "temperature": self.completion_params.temperature,
            "top_p": self.completion_params.top_p,
            "top_k": self.completion_params.top_k,
        }

        if self.completion_params.tools:
            args["tools"] = self.completion_params.tools

        return {k: v for k, v in args.items() if v is not None}

    def _format_request(self, messages: List[Notion]) -> List[dict]:
        """Format messages for Anthropic's API."""
        formatted_messages = []
        for msg in messages:
            msg_content = ""
            try:
                msg_content = json.loads(msg.content)
            except json.decoder.JSONDecodeError:
                if msg.content != "":
                    msg_content = msg.content

            if msg.chat_role == ChatRole.AI:
                if (
                    isinstance(msg_content, list)
                    and len(msg_content) > 0
                    and isinstance(msg_content[0], dict)
                    and "function" in msg_content[0]
                ):
                    # Handle tool calls
                    tool_calls = msg_content
                    formatted_messages.append(
                        {
                            "role": str(AnthropicChatRole.TOOL_CALL.value),
                            "content": json.dumps(tool_calls),
                        }
                    )
                else:
                    formatted_messages.append(
                        {
                            "role": str(AnthropicChatRole.AI.value),
                            "content": msg_content,
                        }
                    )
            else:
                formatted_messages.append(
                    {
                        "role": str(
                            getattr(AnthropicChatRole, msg.chat_role.name).value
                        ),
                        "content": msg_content,
                    }
                )

        return formatted_messages

    def _standardize_response(
        self, response: Union[Message, MessageStreamEvent]
    ) -> List[Notion]:
        """Standardize Anthropic's response format."""
        output: List[Notion] = []

        if isinstance(response, MessageStreamEvent):
            # Streaming response
            if response.type == "content_block_delta":
                if hasattr(response.delta, "text"):
                    output.append(
                        Notion(content=response.delta.text, role=str(ChatRole.AI.value))
                    )
            elif response.type == "input_json_delta":
                output.append(
                    Notion(
                        content=response.delta.partial_json,
                        role=str(ChatRole.TOOL_CALL.value),
                    )
                )
        else:
            # Standard response
            if response.content and len(response.content) > 0:
                output.append(
                    Notion(
                        content=response.content[0].text, role=str(ChatRole.AI.value)
                    )
                )

        return output

    def _postprocess(self, response: List[Notion]) -> List[Notion]:
        """Post-process the response."""
        return response

    def _retry_call(
        self,
        input: List[dict],
        e: Exception,
        api_call: Callable,
        retries: int = 0,
    ):
        """Retry logic for API calls."""
        if not isinstance(input, list):
            raise ValueError("Input must be a list of message dicts.")

        if self.type not in [ModelType.CHAT]:
            raise ValueError(
                f"Invalid model type: {self.type}. " + "Only chat models are supported."
            )

        inp = input.copy()
        # Remove everything since the last user message
        for i in range(len(inp) - 1, -1, -1):
            if inp[i]["role"] == str(AnthropicChatRole.HUMAN.value):
                inp = inp[: i + 1]
                break

        # Inform the AI
        inp.append(
            {
                "role": str(AnthropicChatRole.SYSTEM.value),
                "content": f"Error calling Anthropic API: {e}. "
                + "Do not try to repeat the last action.",
            }
        )

        # Try again
        retries += 1
        return self._common_call_logic(inp, api_call, retries)

    def generate(
        self,
        messages: Messages,
        create_params: Optional[dict] = None,
    ):
        """Generate a response without streaming."""
        create_params = create_params or {}
        return self._common_generate_logic(
            messages,
            False,
            **create_params,
            **self.__chat_args,
        )

    async def agenerate(
        self,
        messages: Messages,
        create_params: Optional[dict] = None,
    ):
        """Generate a response without streaming (async)."""
        create_params = create_params or {}
        return await self._common_generate_logic(
            messages,
            True,
            **create_params,
            **self.__chat_args,
        )

    def stream(
        self,
        messages: Messages,
        create_params: Optional[dict] = None,
    ):
        """Generate a response with streaming."""
        create_params = create_params or {}

        input = self._common_stream_logic(messages)
        output_stream = self._call(
            input, **{**create_params, **self.__chat_args, "stream": True}
        )

        for chunk in output_stream:
            standardized_response = self._postprocess(self._standardize_response(chunk))
            for notion in standardized_response:
                yield notion

    async def astream(
        self,
        messages: Messages,
        create_params: Optional[dict] = None,
    ):
        """Generate a response with streaming (async)."""
        create_params = create_params or {}

        input = self._common_stream_logic(messages)
        output_stream = await self._acall(
            input, **{**create_params, **self.__chat_args, "stream": True}
        )

        async for chunk in output_stream:
            standardized_response = self._postprocess(self._standardize_response(chunk))
            for notion in standardized_response:
                yield notion

    def __init__(
        self,
        name: AnthropicModelName,
        api_key: Optional[str] = None,
        completion_params: Optional[CompletionParams] = None,
    ):
        """
        Creates a new Anthropic model.

        Args:
            name: The name of the Anthropic model to use
            api_key: The API key to use. If None, will attempt to use os.getenv("ANTHROPIC_API_KEY")
            completion_params: The completion parameters to use
        """
        completion_params = completion_params or CompletionParams()

        # For now, we'll use a simple tokenizer since Anthropic doesn't provide one
        # This should be replaced with a proper tokenizer when available
        tokenizer = Tokenizer(
            encode=lambda x: [ord(c) for c in x],
            decode=lambda x: "".join(chr(c) for c in x),
        )

        args = {
            "name": name,
            "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
            "completion_params": completion_params,
            "client": None,
            "client_async": None,
            "llm": None,
            "llm_async": None,
            "can_stream": True,
            "type": ModelType.CHAT,
            "tokenizer": tokenizer,
        }

        if args["api_key"] is None:
            raise ValueError(
                "No Anthropic API key provided and "
                + "`ANTHROPIC_API_KEY` not set as an environment variable."
            )

        if completion_params.max_tokens is not None:
            args["max_response"] = completion_params.max_tokens

        args["client"] = Anthropic(api_key=args["api_key"])
        args["client_async"] = AsyncAnthropic(api_key=args["api_key"])
        args["llm"] = args["client"].messages.create
        args["llm_async"] = args["client_async"].messages.create

        super().__init__(**args)
