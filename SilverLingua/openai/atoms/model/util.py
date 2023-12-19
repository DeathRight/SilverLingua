from typing import List, Literal, Optional, Union

from openai.types.chat import (
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel, Field

OpenAIChatModels = {
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo-1106": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
}
"""
    List of OpenAI chat models and their maximum number of tokens.
"""

OpenAIChatModelName = Literal[
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
]

OpenAIEmbeddingModels = {"text-embedding-ada-002": 8191}
"""
    List of OpenAI embedding models and their maximum number of tokens.
"""

OpenAIEmbeddingModelName = Literal["text-embedding-ada-002",]

##########
OpenAIModels = {**OpenAIChatModels, **OpenAIEmbeddingModels}
"""
    List of OpenAI models and their maximum number of tokens.

    See: `OpenAIChatModels` and `OpenAIEmbeddingModels`
"""

OpenAIModelName = Union[OpenAIChatModelName, OpenAIEmbeddingModelName]


class CompletionParams(BaseModel):
    """
    Optional parameters used when calling the OpenAI completions API.
    """

    tools: Optional[List[ChatCompletionToolParam]] = Field(
        default=None, description="List of tools to bind to the chat session."
    )

    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = Field(
        default=None,
        description="Controls which (if any) function is called by the model.",
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
        ge=1.0,
    )

    seed: Optional[int] = Field(
        default=None,
        description="The seed to use for deterministic sampling.",
    )

    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description="The format of the response.",
    )

    top_p: float = Field(
        default=1.0,
        description="The nucleus sampling probability.",
    )
    temperature: float = Field(
        default=1.0,
        description="The sampling temperature.",
    )
    n: int = Field(
        default=1,
        description="The number of completions to generate.",
    )
    stop: List[str] = Field(
        default_factory=list,
        description="List of tokens to stop completion at.",
    )
    presence_penalty: float = Field(
        default=0.0,
        description="The presence penalty.",
    )
    frequency_penalty: float = Field(
        default=0.0,
        description="The frequency penalty.",
    )
    logit_bias: dict = Field(
        default_factory=dict,
        description="The logit bias.",
    )

    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user.",
    )
