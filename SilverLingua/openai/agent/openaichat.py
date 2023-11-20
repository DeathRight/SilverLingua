from typing import List

from SilverLingua.core.agent import Agent
from SilverLingua.core.atoms import Idearium, Tool

from ..atoms import OpenAIModel
from ..atoms.model import ChatCompletionInputTool


class OpenAIChatAgent(Agent):
    """
    An agent that uses the OpenAI chat completion API.
    """

    _model: OpenAIModel

    @property
    def model(self) -> OpenAIModel:
        return self._model

    def _bind_tools(self) -> None:
        m_tools: List[ChatCompletionInputTool] = [
            ChatCompletionInputTool(tool.description) for tool in self.tools
        ]

        self._model.tools = m_tools

    def __init__(
        self,
        model_name: str,
        idearium: Idearium = None,
        tools: List[Tool] = None,
        **kwargs
    ) -> None:
        """
        Initializes the OpenAI chat agent.

        Args:
            model_name (str): The name of the OpenAI model to use.
            idearium (Idearium): The idearium to use.
            If None, a new one will be created.
            tools (List[Tool]): The tools to use.
            If None, no tools will be used.
            **kwargs: Additional keyword arguments to pass to the
            OpenAIModel constructor.
        """
        self._model = OpenAIModel(name=model_name, **kwargs)
        super().__init__(model=self._model, idearium=idearium, tools=tools)
