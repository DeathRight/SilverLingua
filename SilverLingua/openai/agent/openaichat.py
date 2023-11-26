import logging
from typing import List, Optional

from openai.types.chat import (
    ChatCompletionToolParam,
)

from SilverLingua.core.agent import Agent
from SilverLingua.core.atoms import Idearium, Tool

from ..atoms import OpenAIModel
from ..atoms.model.util import CompletionParams, OpenAIChatModelName

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
            self.model.tools = m_tools

    def __init__(
        self,
        model_name: OpenAIChatModelName = "gpt-3.5-turbo",
        idearium: Optional[Idearium] = None,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None,
        completion_params: Optional[CompletionParams] = None,
    ) -> None:
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
        self.model = OpenAIModel(
            name=model_name, api_key=api_key, completion_params=completion_params
        )

        super().__init__(model=self.model, idearium=idearium, tools=tools)
