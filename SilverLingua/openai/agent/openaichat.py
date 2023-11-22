from typing import List

from SilverLingua.core.agent import Agent
from SilverLingua.core.atoms import Idearium, Notion, Tool

from ..atoms import OpenAIModel
from ..atoms.model import ChatCompletionInputTool
from ..atoms.model.util import (
    ChatCompletionInputMessageToolResponse,
    ChatCompletionMessageToolCalls,
)


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
            ChatCompletionInputTool(tool.description).to_dict() for tool in self.tools
        ]

        # Check to make sure m_tools is not empty
        if len(m_tools) > 0:
            self._model.tools = m_tools

    def _use_tool(self, notion: Notion) -> List[Notion]:
        responses: List[ChatCompletionInputMessageToolResponse] = []

        tool_calls = ChatCompletionMessageToolCalls.from_json(notion.content)
        for tool_call in tool_calls:
            tool = self._find_tool(tool_call.function.name)
            if tool is not None:
                responses.append(
                    ChatCompletionInputMessageToolResponse(
                        tool_call.id,
                        tool_call.function.name,
                        tool(tool_call.function),
                    )
                )
            else:
                responses.append(
                    ChatCompletionInputMessageToolResponse(
                        tool_call.id,
                        "error",
                        "Tool not found",
                    )
                )

        return [
            Notion(response.to_json(), str(self.role.TOOL_RESPONSE.value))
            for response in responses
        ]

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
