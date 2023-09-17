import os

import openai
import tiktoken

from SilverLingua.core.model import Model

# List of OpenAI models and their maximum number of tokens.
OpenAIModels = {
    "text-davinci-003": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
}


class OpenAI(Model):
    """
    An OpenAI model.
    """

    temperature: float
    top_p: float
    n: int
    stop: list
    presence_penalty: float
    frequency_penalty: float
    logit_bias: dict

    @property
    def api_key(self) -> str:
        """
        The API key for the model.

        Defaults to the OPENAI_API_KEY environment variable.
        """
        return self._api_key

    @property
    def name(self) -> str:
        """
        The name of the model version being used.
        """
        return self._name

    @property
    def model(self) -> object:
        """
        The model itself.
        """
        return self._model

    @property
    def can_stream(self) -> bool:
        """
        Whether the model can be streamed.
        """
        return self._can_stream

    @property
    def streaming(self) -> bool:
        """
        Whether the model is initialized as streaming.
        """
        return self._streaming

    @property
    def max_tokens(self) -> int:
        """
        The maximum number of tokens the model can handle.
        """
        return OpenAIModels[self.name]

    @property
    def max_response(self) -> int:
        """
        The maximum number of tokens the model can return.

        If 0, there is no limit other than the maximum number of tokens for the model.
        """
        return self._max_response

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        """
        The tokenizer for the model.
        """
        return tiktoken.encoding_for_model(self.name)

    def __init__(
        self,
        name: str = "gpt-3.5-turbo",
        streaming: bool = False,
        max_response: int = 256,
        api_key: str = os.getenv["OPENAI_API_KEY"],
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
            name (str, optional): The name of the model version being used. Defaults to "gpt-3.5-turbo".
            streaming (bool, optional): Whether the model should be initialized as streaming. Defaults to False.
            max_response (int, optional): The maximum number of tokens the model can return. Defaults to 256.
            api_key (str, optional): The API key for the model. Defaults to the OPENAI_API_KEY environment variable.
        """
        self._api_key = api_key
        openai.api_key = self.api_key

        if name not in OpenAIModels:
            raise ValueError(f"Invalid OpenAI model name: {name}")
        self._name = name

        if self._name == "text-davinci-003":
            self._can_stream = False
            self._model = openai.Completion
        else:
            self._can_stream = True
            self._model = openai.ChatCompletion

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

    # TODO: Implement generation
