import logging
from typing import List, Optional

from openai.types.chat import (
    ChatCompletionToolParam,
)

from SilverLingua.core.atoms import Tool
from SilverLingua.core.organisms import Idearium
from SilverLingua.core.templates.agent import Agent

from ..model import OpenAIModel
from ..model.util import CompletionParams, OpenAIChatModelName

logger = logging.getLogger(__name__)


class OpenAIChatAgent(Agent):
    """
    An agent that uses the OpenAI chat completion API.
    """

    model: OpenAIModel

    @property
    def model(self) -> OpenAIModel:
        return self._model

    def _bind_tools(self) -> None:
        m_tools: List[ChatCompletionToolParam] = [
            {"type": "function", "function": tool.description} for tool in self.tools
        ]

        # Check to make sure m_tools is not empty
        if len(m_tools) > 0:
            self.model.completion_params.tools = m_tools

    def __init__(
        self,
        model_name: OpenAIChatModelName = "gpt-3.5-turbo",
        idearium: Optional[Idearium] = None,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None,
        completion_params: Optional[CompletionParams] = None,
    ):
        """
        Initializes the OpenAI chat agent.

        Args:
            model_name (str, optional): The name of the OpenAI model to use.
            Default is "gpt-3.5-turbo".
            idearium (Idearium, optional): The idearium to use.
            If None, a new one will be created.
            tools (List[Tool], optional): The tools to use.
            If None, no tools will be used.
            api_key (str, optional): The OpenAI API key to use.
            If None, will attempt to use os.getenv("OPENAI_API_KEY").
            completion_params (CompletionParams, optional): The completion parameters
            to use.
        """
        model = OpenAIModel(
            name=model_name, api_key=api_key, completion_params=completion_params
        )
        # print(f"Testing 2: Tokenizer: {model.tokenizer}")

        args = {"model": model}
        if idearium is not None:
            args["idearium"] = idearium
        if tools is not None:
            args["tools"] = tools

        super().__init__(**args)
