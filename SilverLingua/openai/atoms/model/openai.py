import os
from typing import List, Optional, Union

import openai
import tiktoken

from SilverLingua.core.atoms.memory import Notion
from SilverLingua.core.atoms.model import Model, ModelType
from SilverLingua.core.atoms.role import ChatRole, OpenAIChatRole

from ... import logger
from .util import ChatCompletionInputMessage

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

    # Completion parameters
    temperature: float
    top_p: float
    n: int
    stop: list
    presence_penalty: float
    frequency_penalty: float
    logit_bias: dict

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
    def moderation(self) -> openai.Moderation:
        """
        The moderation object used to check if text violates OpenAI's content policy.
        """
        return openai.Moderation

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
    def chat_args(self):
        """
        Boilerplate arguments for the OpenAI chat completion API to be unpacked.
        """
        return {
            "model": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.streaming,
            "stop": self.stop,
            "max_tokens": self.max_response,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
        }

    @property
    def text_args(self):
        """
        Boilerplate arguments for the OpenAI text completion API to be unpacked.
        """
        return {
            "model": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": self.stop,
            "max_tokens": self.max_response,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
        }

    def _preprocess(self, messages: List[Notion]):
        return messages

    def _trim(self, messages: List[Notion]):
        return_messages = messages.copy()
        available_tokens = self.max_tokens

        system_msg = None
        system_msg_index = None

        user_msg_count = 0
        last_user_index = 0

        # Check if the number of messages is greater than the maximum number of tokens.
        if len(return_messages) > self.max_tokens:
            # Trim the messages to the maximum number of tokens.
            return_messages = return_messages[: self.max_tokens]

        # Tokenize the messages.
        rm = return_messages.copy()
        tokenized_msgs = []
        i = -1
        for msg in return_messages:
            i += 1
            t_msg = self.tokenizer.encode(msg.content)

            # If the message is a system message, save it for later,
            # remove it from the list of messages, and subtract its length
            # from the available tokens. This is because system messages
            # should be protected from being trimmed.
            if msg.role == ChatRole.SYSTEM:
                system_msg = msg
                system_msg_index = i
                available_tokens -= len(t_msg)
                rm.pop(i)
                i -= 1  # To keep the index in sync
                continue
            elif msg.role == ChatRole.USER:
                user_msg_count += 1
                last_user_index = i

            available_tokens -= len(t_msg)
            tokenized_msgs.append(t_msg)
        return_messages = rm

        # If there is only one message
        if len(tokenized_msgs) == 1:
            # If available tokens is less than 0, trim the message
            # to the available tokens.
            if available_tokens < 0:
                tokenized_msgs[0] = tokenized_msgs[0][:available_tokens]
                return_messages[0].content = self.tokenizer.decode(tokenized_msgs[0])

            return return_messages
        elif len(tokenized_msgs) > 1 and available_tokens < 0:
            if user_msg_count == 1:
                # If there is only one user message, trim the message
                # to the available tokens.
                tokenized_msgs[last_user_index] = tokenized_msgs[last_user_index][
                    :available_tokens
                ]
                return_messages[last_user_index].content = self.tokenizer.decode(
                    tokenized_msgs[last_user_index]
                )
            # If available tokens is less than 0, trim entire messages
            # starting from the top until available tokens is 0 or greater.
            for i in range(len(tokenized_msgs)):
                if available_tokens >= 0:
                    break

                available_tokens += len(tokenized_msgs[i])
                tokenized_msgs.pop(i)
                return_messages.pop(i)

        # Place system message back into the list of messages.
        if system_msg is not None:
            return_messages.insert(system_msg_index, system_msg)
        return return_messages

    def _call(
        self,
        input: Union[str, List[ChatCompletionInputMessage], List[str], List[List[int]]],
        **kwargs,
    ):
        if input is None:
            raise ValueError("No input provided.")

        if self.type == ModelType.TEXT:
            if input is not str:
                raise ValueError("Input must be a string.")
            return self.model.create(**self.text_args, prompt=input)
        elif self.type == ModelType.CHAT:
            if input is not list:
                raise ValueError("Input must be a list of ChatCompletionInputMessage.")
            return self.model.create(**self.chat_args, messages=input, **kwargs)
        elif self.type == ModelType.EMBEDDING:
            # TODO: Implement
            pass

    def generate(self, messages: List[Notion], **kwargs) -> str:
        if messages is None:
            raise ValueError("No messages provided.")

        input = self._formatter(self._preprocess(self._trim(messages)))

        output = self._call(input, **kwargs)

        return self._postprocess(output)

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
        suffix: Optional[str] = None,
        logprobs: Optional[int] = None,
        echo: bool = False,
        best_of: int = 1,
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
            suffix (str, optional): The suffix for the model. Defaults to None.
            logprobs (int, optional): The number of logprobs for the model.
            Defaults to None.
            echo (bool, optional): Echo back the prompt in addition to the completion.
            Defaults to False.
            best_of (int, optional): Generates `best_of` completions server-side and
            returns the "best" (the one with the lowest log probability per token).
            Defaults to 1.
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

        # Text completion specific parameters
        self.suffix = suffix
        self.logprobs = logprobs
        self.echo = echo
        self.best_of = best_of
