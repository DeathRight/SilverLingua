import os
from typing import List

import openai
import tiktoken

from SilverLingua.core.atoms.memory import Notion
from SilverLingua.core.atoms.model import Model, ModelType
from SilverLingua.core.atoms.role import ChatRole, OpenAIChatRole

from ... import logger

# List of OpenAI models and their maximum number of tokens.
OpenAIModels = {
    "text-embedding-ada-002": 8191,
    "text-davinci-003": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
}


class OpenAIModel(Model):
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
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> ModelType:
        return self._type

    @property
    def model(self) -> object:
        return self._model

    @property
    def can_stream(self) -> bool:
        return self._can_stream

    @property
    def streaming(self) -> bool:
        return self._streaming

    @property
    def max_tokens(self) -> int:
        return OpenAIModels[self.name]

    @property
    def max_response(self) -> int:
        return self._max_response

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self.name)

    def generate(self, messages: List[Notion]) -> str:
        if messages is None:
            raise ValueError("No messages provided.")

        msgs = self._preprocess(self._trim(messages))

        self._formatter(msgs)

        # TODO: Implement generation

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
            name (str, optional): The name of the model version being used.
            Defaults to "gpt-3.5-turbo".
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
        """
        self._api_key = api_key
        openai.api_key = self.api_key

        if name not in OpenAIModels:
            raise ValueError(f"Invalid OpenAI model name: {name}")
        self._name = name

        if self._name == "text-davinci-003":
            self._can_stream = False
            self._model = openai.Completion
            self._type = ModelType.TEXT
            logger.warning(
                "The text-davinci-003 will be deprecated 2024-01-04. (https://openai.com/blog/gpt-4-api-general-availability)"
            )
        elif self._name == "text-embedding-ada-002":
            self._can_stream = False
            self._model = openai.Embedding
            self._type = ModelType.EMBEDDING
        else:
            self._can_stream = True
            self._model = openai.ChatCompletion
            self._type = ModelType.CHAT

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
