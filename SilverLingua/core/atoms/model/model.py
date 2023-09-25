from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union

from ..memory import Notion
from ..role import ChatRole


class ModelType(Enum):
    CHAT = 0
    TEXT = 1
    EMBEDDING = 2
    CODE = 3


class Model(ABC):
    """
    Abstract class for all Large Language Models.
    """

    @property
    @abstractmethod
    def role(self) -> ChatRole:
        """
        The ChatRole object for the model.
        """
        pass

    @property
    @abstractmethod
    def api_key(self) -> str:
        """
        The API key for the model.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the model version being used.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> ModelType:
        """
        The type of model being used.
        """
        pass

    @property
    @abstractmethod
    def model(self) -> object:
        """
        The model itself.
        """
        pass

    @property
    @abstractmethod
    def can_stream(self) -> bool:
        """
        Whether the model can be streamed.
        """
        pass

    @property
    @abstractmethod
    def streaming(self) -> bool:
        """
        Whether the model is initialized as streaming.
        """
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """
        The maximum number of tokens the model can handle.
        """
        pass

    @property
    @abstractmethod
    def max_response(self) -> int:
        """
        The maximum number of tokens the model can return.

        If 0, there is no limit.
        """
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> object:
        """
        The tokenizer for the model.
        """
        pass

    @abstractmethod
    def _trim(self, messages: List[Notion], *args, **kwargs) -> List[Notion]:
        """
        Trims the List of `Notions` to fit the maximum number of tokens for the model.

        This is a lifecycle method that is called by the `generate` method.

        Lifecycle:
            1. `generate` calls `_trim` with the given List of `Notions`.
            2. `_trim` returns the trimmed List of `Notions` to be passed
            to `_preprocess`.
            3. `_preprocess` returns the List of `Notions` to be passed to `_formatter`.
            4. `_formatter` formats the List of `Notions` into a prompt or object that
            can be passed to the model.
            5. `_call` calls the model with the prompt or object
            and returns the response.
            6. `generate` calls `_postprocess` with the response.
            7. `_postprocess` returns a str response to be returned by `generate`.
        """

    @abstractmethod
    def _preprocess(self, messages: List[Notion], *args, **kwargs) -> List[Notion]:
        """
        Preprocesses the List of `Notions` before passing it to the `_formatter`.

        This is a lifecycle method that is called by the `generate` method.

        Lifecycle:
            1. `generate` calls `_trim` with the given List of `Notions`.
            2. `_trim` returns the trimmed List of `Notions` to be passed
            to `_preprocess`.
            3. `_preprocess` returns the List of `Notions` to be passed to `_formatter`.
            4. `_formatter` formats the List of `Notions` into a prompt or object that
            can be passed to the model.
            5. `_call` calls the model with the prompt or object
            and returns the response.
            6. `generate` calls `_postprocess` with the response.
            7. `_postprocess` returns a str response to be returned by `generate`.
        """
        pass

    @abstractmethod
    def _formatter(self, messages: List[Notion], *args, **kwargs) -> Union[str, object]:
        """
        Formats the List of `Notions` into a prompt or object
        that can be passed to the model.

        This is a lifecycle method that is called by the `generate` method.

        Lifecycle:
            1. `generate` calls `_trim` with the given List of `Notions`.
            2. `_trim` returns the trimmed List of `Notions` to be passed
            to `_preprocess`.
            3. `_preprocess` returns the List of `Notions` to be passed to `_formatter`.
            4. `_formatter` formats the List of `Notions` into a prompt or object that
            can be passed to the model.
            5. `_call` calls the model with the prompt or object
            and returns the response.
            6. `generate` calls `_postprocess` with the response.
            7. `_postprocess` returns a str response to be returned by `generate`.
        """
        pass

    @abstractmethod
    def _postprocess(self, response: Union[object, str], *args, **kwargs) -> str:
        """
        Postprocesses the response from the model.

        This is a lifecycle method that is called by the `generate` method.

        Lifecycle:
            1. `generate` calls `_trim` with the given List of `Notions`.
            2. `_trim` returns the trimmed List of `Notions` to be passed
            to `_preprocess`.
            3. `_preprocess` returns the List of `Notions` to be passed to `_formatter`.
            4. `_formatter` formats the List of `Notions` into a prompt or object that
            can be passed to the model.
            5. `_call` calls the model with the prompt or object
            and returns the response.
            6. `generate` calls `_postprocess` with the response.
            7. `_postprocess` returns a str response to be returned by `generate`.
        """
        pass

    @abstractmethod
    def _call(self, input: Union[str, object], *args, **kwargs) -> object:
        """
        Calls the model with the given input and returns the response.

        Should behave exactly as `_acall` does, but synchronously.

        This is a lifecycle method that is called by the `generate` method.

        Lifecycle:
            1. `generate` calls `_trim` with the given List of `Notions`.
            2. `_trim` returns the trimmed List of `Notions` to be passed
            to `_preprocess`.
            3. `_preprocess` returns the List of `Notions` to be passed to `_formatter`.
            4. `_formatter` formats the List of `Notions` into a prompt or object that
            can be passed to the model.
            5. `_call` calls the model with the prompt or object
            and returns the response.
            6. `generate` calls `_postprocess` with the response.
            7. `_postprocess` returns a str response to be returned by `generate`.
        """
        pass

    @abstractmethod
    async def _acall(self, input: Union[str, object], *args, **kwargs) -> object:
        """
        Calls the model with the given input and returns the response asynchronously.

        Should behave exactly as `_call` does, but asynchronously.

        This is a lifecycle method that is called by the `agenerate` method.

        Lifecycle:
            1. `agenerate` calls `_trim` with the given List of `Notions`.
            2. `_trim` returns the trimmed List of `Notions` to be passed
            to `_preprocess`.
            3. `_preprocess` returns the List of `Notions` to be passed to `_formatter`.
            4. `_formatter` formats the List of `Notions` into a prompt or object that
            can be passed to the model.
            5. `_acall` calls the model with the prompt or object
            and returns the response.
            6. `agenerate` calls `_postprocess` with the response.
            7. `_postprocess` returns a str response to be returned by `agenerate`.
        """

    @abstractmethod
    def generate(self, messages: List[Notion], *args, **kwargs) -> str:
        """
        Calls the model with the given List of `Notions` and returns the response.

        Should behave exactly as `agenerate` does, but synchronously.

        This is the primary method for generating responses from the model,
        and is responsible for calling all of the lifecycle methods.
        """
        pass

    @abstractmethod
    async def agenerate(self, messages: List[Notion], *args, **kwargs) -> str:
        """
        Calls the model with the given List of `Notions` and returns the response
        asynchronously.

        Should behave exactly as `generate` does, but asynchronously.

        This is the primary method for generating async responses from the model,
        and is responsible for calling all of the lifecycle methods.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, *args, **kwargs) -> object:
        """
        Streams the model with the given prompt using the USER role and
        returns the response.

        If the model cannot be streamed, this will raise an exception.
        """
        pass

    @abstractmethod
    def __init__(self, streaming: bool = False, max_response: int = 0, **kwargs):
        """
        Initializes the model.
        """
        pass
