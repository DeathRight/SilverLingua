from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generator, List, Union

from pydantic import BaseModel, ConfigDict, Field

from ..memory import Idearium, Notion, Tokenizer
from ..role import ChatRole


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
    1. Pre-processing (_preprocess): Performs any necessary transformations or
        adjustments to the messages prior to trimming or preparing them for model input.
        (Optional)

    2. Preparing Request (_format_request): Converts the pre-processed messages
        into a format suitable for model input.

    3. Model Invocation (_call or _acall): Feeds the prepared input to the LLM and
        retrieves the raw model output. There should be both synchronous and
        asynchronous versions available.

    4. Standardizing Response (_standardize_response): Transforms the raw model
        output into a consistent response format suitable for further processing or
        delivery.

    5. Post-processing (_postprocess): Performs any final transformations or
        adjustments to the standardized responses, making them ready for delivery.
        (Optional)

    Subclasses should implement each of the non-optional lifecycle steps in accordance
    with the specific requirements and behaviors of the target LLM.
    """

    model_config = ConfigDict(frozen=True)
    #
    max_response: int = Field(default=0)
    api_key: str
    name: str
    #
    role: ChatRole
    type: ModelType
    llm: object
    llm_async: object
    can_stream: bool
    tokenizer: Tokenizer

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """
        The maximum number of tokens that can be fed to the model at once.
        """
        pass

    @abstractmethod
    def _preprocess(self, messages: List[Notion], *args, **kwargs) -> List[Notion]:
        """
        Preprocesses the List of `Notions`, applying any effects necessary
        before being prepped for input into an API.

        This is a lifecycle method that is called by the `generate` method.
        """
        pass

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
    def _call(self, input: Union[str, object, List[any]], *args, **kwargs) -> object:
        """
        Calls the model with the given input and returns the raw response.

        Should behave exactly as `_acall` does, but synchronously.

        This is a lifecycle method that is called by the `generate` method.
        """
        pass

    @abstractmethod
    async def _acall(
        self, input: Union[str, object, List[any]], *args, **kwargs
    ) -> object:
        """
        Calls the model with the given input and returns the
        raw response asynchronously.

        Should behave exactly as `_call` does, but asynchronously.

        This is a lifecycle method that is called by the `agenerate` method.
        """
        pass

    @abstractmethod
    def generate(
        self, messages: Union[Idearium, List[Notion]], *args, **kwargs
    ) -> List[Notion]:
        """
        Calls the model with the given messages and returns the response.

        Messages can be either an Idearium or a List of Notions.

        This is the primary method for generating responses from the model,
        and is responsible for calling all of the lifecycle methods.
        """
        pass

    @abstractmethod
    async def agenerate(
        self, messages: Union[Idearium, List[Notion]], *args, **kwargs
    ) -> List[Notion]:
        """
        Calls the model with the given messages and returns the response
        asynchronously.

        Messages can be either an Idearium or a List of Notions.

        This is the primary method for generating async responses from the model,
        and is responsible for calling all of the lifecycle methods.
        """
        pass

    @abstractmethod
    def stream(
        self, messages: Union[Idearium, List[Notion]], *args, **kwargs
    ) -> Generator[Notion, Any, None]:
        """
        Streams the model with the given messages and returns the response,
        one token at a time.

        If the model cannot be streamed, this will raise an exception.
        """
        pass

    @abstractmethod
    async def astream(
        self, messages: Union[Idearium, List[Notion]], *args, **kwargs
    ) -> Generator[Notion, Any, None]:
        """
        Streams the model with the given messages and returns the response,
        one token at a time, asynchronously.

        If the model cannot be streamed, this will raise an exception.
        """
        pass
