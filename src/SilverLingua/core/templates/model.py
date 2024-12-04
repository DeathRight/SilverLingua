import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Generator, List, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from ..atoms import ChatRole, Tokenizer
from ..molecules import Notion
from ..organisms import Idearium

logger = logging.getLogger(__name__)

Messages = Union[str, Notion, Idearium, List[Union[str, Notion]]]
"""
A type alias for the various types of messages that can be passed to a model.
"""


class ModelType(Enum):
    CHAT = 0
    EMBEDDING = 1
    CODE = 2


class Model(BaseModel, ABC):
    """
    Abstract class for all Large Language Models.

    This class outlines a standardized lifecycle for interacting with LLMs,
    aimed at ensuring a consistent process for message trimming, pre-processing,
    preparing requests for the model, invoking the model, standardizing the response,
    and post-processing. The lifecycle is as follows:

    Lifecycle:
        1. **Pre-processing ([`_preprocess`][silverlingua.core.templates.model.Model._preprocess]):** Performs any necessary transformations or
            adjustments to the messages prior to trimming or preparing them for model input.
            <span style="color:var(--md-accent-fg-color)">*(Optional)*</span>

        2. **Preparing Request ([`_format_request`][silverlingua.core.templates.model.Model._format_request]):** Converts the pre-processed messages
            into a format suitable for model input.

        3. **Model Invocation ([`_call`][silverlingua.core.templates.model.Model._call] or [`_acall`][silverlingua.core.templates.model.Model._acall]):** Feeds the prepared input to the LLM and
            retrieves the raw model output. There should be both synchronous and
            asynchronous versions available.

        4. **Standardizing Response ([`_standardize_response`][silverlingua.core.templates.model.Model._standardize_response]):** Transforms the raw model
            output into a consistent response format suitable for further processing or
            delivery.

        5. **Post-processing ([`_postprocess`][silverlingua.core.templates.model.Model._postprocess]):** Performs any final transformations or
            adjustments to the standardized responses, making them ready for delivery.
            <span style="color:var(--md-accent-fg-color)">*(Optional)*</span>

    Subclasses should implement each of the non-optional lifecycle steps in accordance
    with the specific requirements and behaviors of the target LLM.

    See also:
        - [`Agent`][silverlingua.core.templates.agent.Agent]
        - [`Idearium`][silverlingua.core.organisms.idearium.Idearium]
        - [`Notion`][silverlingua.core.molecules.notion.Notion]
    """

    model_config = ConfigDict(frozen=True)
    #
    max_response: int = Field(default=0)
    api_key: str
    name: str
    #
    role: Type[ChatRole]
    type: ModelType
    llm: Callable
    llm_async: Callable
    can_stream: bool
    tokenizer: Tokenizer

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """
        The maximum number of tokens that can be fed to the model at once.
        """
        pass

    def _process_input(self, messages: Messages) -> Idearium:
        if isinstance(messages, str):
            notions = [Notion(content=messages, role=self.role.HUMAN)]
        elif isinstance(messages, Notion):
            notions = [messages]
        elif isinstance(messages, Idearium):
            return messages  # Already an Idearium, no need to convert
        elif isinstance(messages, list):
            notions = [
                (
                    Notion(content=msg, role=self.role.HUMAN)
                    if isinstance(msg, str)
                    else msg
                )
                for msg in messages
            ]
        else:
            raise ValueError("Invalid input type for messages")

        return Idearium(self.tokenizer, self.max_tokens, notions)

    def _convert_role(self, role: ChatRole) -> str:
        """
        Converts the standard ChatRole to the model-specific role.
        """
        return str(self.role[role.name].value)

    def _preprocess(self, messages: List[Notion]) -> List[Notion]:
        """
        Preprocesses the List of `Notions`, applying any effects necessary
        before being prepped for input into an API.

        This is a lifecycle method that is called by the `generate` method.
        """
        return [
            Notion(msg.content, self._convert_role(msg.chat_role), msg.persistent)
            for msg in messages
        ]

    @abstractmethod
    def _format_request(
        self, messages: List[Notion], *args, **kwargs
    ) -> Union[str, object]:
        """
        Formats the List of `Notions` into a format suitable for model input.

        This is a lifecycle method that is called by the `generate` method.
        """
        pass

    @abstractmethod
    def _standardize_response(
        self, response: Union[object, str, List[any]], *args, **kwargs
    ) -> List[Notion]:
        """
        Standardizes the raw response from the model into a List of Notions.

        This is a lifecycle method that is called by the `generate` method.
        """
        pass

    @abstractmethod
    def _postprocess(self, response: List[Notion], *args, **kwargs) -> List[Notion]:
        """
        Postprocesses the response from the model, applying any final effects
        before being returned.

        This is a lifecycle method that is called by the `generate` method.
        """
        pass

    @abstractmethod
    def _retry_call(
        self,
        input: Union[str, object, List[any]],
        e: Exception,
        api_call: Callable,
        retries: int = 0,
    ) -> Union[str, object]:
        """
        Retry logic for API calls used by `_common_call_logic`.
        """
        pass

    def _common_call_logic(
        self,
        input: Union[str, object, List[any]],
        api_call: Callable,
        retries: int = 0,
    ) -> Union[str, object]:
        if input is None:
            raise ValueError("No input provided.")

        try:
            out = api_call(messages=input)
            return out
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            if retries >= 3:
                raise e

            return self._retry_call(input, e, api_call, retries=retries)

    def _call(
        self, input: Union[str, object, List[any]], retries: int = 0, **kwargs
    ) -> object:
        """
        Calls the model with the given input and returns the raw response.

        Should behave exactly as `_acall` does, but synchronously.

        This is a lifecycle method that is called by the `generate` method.
        """

        def api_call(**kwargs_):
            return self.llm(**kwargs_, **kwargs)

        return self._common_call_logic(input, api_call, retries)

    async def _acall(
        self, input: Union[str, object, List[any]], retries: int = 0, **kwargs
    ) -> object:
        """
        Calls the model with the given input and returns the
        raw response asynchronously.

        Should behave exactly as `_call` does, but asynchronously.

        This is a lifecycle method that is called by the `agenerate` method.
        """

        async def api_call(**kwargs_):
            return await self.llm_async(**kwargs_, **kwargs)

        result = self._common_call_logic(input, api_call, retries)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _common_generate_logic(
        self,
        messages: Messages,
        is_async=False,
        **kwargs,
    ):
        if messages is None:
            raise ValueError("No messages provided.")

        call_method = self._acall if is_async else self._call

        idearium = self._process_input(messages)
        input = self._format_request(self._preprocess(idearium))

        if is_async:

            async def call():
                response = await call_method(input, **kwargs)
                output = self._standardize_response(response)
                return self._postprocess(output)

            return call()
        else:
            response = call_method(input, **kwargs)
            output = self._standardize_response(response)
            return self._postprocess(output)

    @abstractmethod
    def generate(
        self,
        messages: Messages,
        *args,
        **kwargs,
    ) -> List[Notion]:
        """
        Calls the model with the given messages and returns the response.

        Messages can be any of:
        string, list of strings, Notion, list of Notions, or Idearium.

        This is the primary method for generating responses from the model,
        and is responsible for calling all of the lifecycle methods.
        """
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: Messages,
        *args,
        **kwargs,
    ) -> List[Notion]:
        """
        Calls the model with the given messages and returns the response
        asynchronously.

        Messages can be any of:
        string, list of strings, Notion, list of Notions, or Idearium.

        This is the primary method for generating async responses from the model,
        and is responsible for calling all of the lifecycle methods.
        """
        pass

    def _common_stream_logic(self, messages: Messages):
        if messages is None:
            raise ValueError("No messages provided.")

        if not self.can_stream:
            raise ValueError(
                "This model does not support streaming. "
                + "Please use the `generate` method instead."
            )

        idearium = self._process_input(messages)
        input = self._format_request(self._preprocess(idearium))
        return input

    @abstractmethod
    def stream(
        self, messages: Messages, *args, **kwargs
    ) -> Generator[Notion, Any, None]:
        """
        Streams the model with the given messages and returns the response,
        one token at a time.

        Messages can be any of:
        string, list of strings, Notion, list of Notions, or Idearium.

        If the model cannot be streamed, this will raise an exception.
        """
        pass

    @abstractmethod
    async def astream(
        self, messages: Messages, *args, **kwargs
    ) -> Generator[Notion, Any, None]:
        """
        Streams the model with the given messages and returns the response,
        one token at a time, asynchronously.

        Messages can be any of:
        string, list of strings, Notion, list of Notions, or Idearium.

        If the model cannot be streamed, this will raise an exception.
        """
        pass
