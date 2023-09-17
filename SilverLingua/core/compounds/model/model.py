from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract class for all Large Language Models.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the model version being used.
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
    def respond(self, prompt: str) -> str:
        """
        Calls the model with the given prompt and returns the response.
        """
        pass

    @abstractmethod
    async def arespond(self, prompt: str) -> str:
        """
        Calls the model asynchronously with the given prompt and returns the response.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str) -> str:
        """
        Streams the model with the given prompt and returns the response.
        """
        pass

    @abstractmethod
    def __init__(self, streaming: bool = False, max_response: int = 0, **kwargs):
        """
        Initializes the model.
        """
        pass
