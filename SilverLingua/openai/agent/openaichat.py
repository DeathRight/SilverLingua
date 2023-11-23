import json
import logging
from typing import List, Optional

from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)

from SilverLingua.core.agent import Agent
from SilverLingua.core.atoms import Idearium, Notion, Tool

from ..atoms import OpenAIModel

logger = logging.getLogger(__name__)


class OpenAIChatAgent(Agent):
    """
    An agent that uses the OpenAI chat completion API.
    """

    _model: OpenAIModel

    @property
    def model(self) -> OpenAIModel:
        return self._model

    def _bind_tools(self) -> None:
        m_tools: List[ChatCompletionToolParam] = [
            {"type": "function", "function": tool.description} for tool in self.tools
        ]

        # Check to make sure m_tools is not empty
        if len(m_tools) > 0:
            self._model.tools = m_tools

    def _use_tool(self, notion: Notion) -> List[Notion]:
        responses: List[ChatCompletionToolMessageParam] = []

        tool_calls: List[ChatCompletionMessageToolCall] = json.loads(notion.content)
        if not isinstance(tool_calls, list):
            raise ValueError("Notion content must be a list of tool calls")

        for t in tool_calls:
            tool = self._find_tool(t["function"]["name"])
            if tool is not None:
                args = None

                # Try to parse the arguments as JSON
                try:
                    args = json.loads(t["function"]["arguments"])
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON in tool call arguments:"
                        + f"{t['function']['arguments']}"
                    )

                if args is not None:
                    responses.append(
                        {
                            "tool_call_id": t["id"],
                            "content": tool(**json.loads(t["function"]["arguments"])),
                            "role": "tool",
                        }
                    )
                else:
                    # Tell the AI that the arguments were invalid
                    responses.append(
                        {
                            "tool_call_id": t["id"],
                            "content": "ERROR: Invalid arguments",
                            "role": "tool",
                        }
                    )
            else:
                responses.append(
                    {
                        "tool_call_id": t["id"],
                        "content": "Tool not found",
                        "role": "tool",
                    }
                )

        return [
            Notion(
                json.dumps(
                    {
                        "tool_call_id": response["tool_call_id"],
                        "content": response["content"],
                    }
                ),
                str(self.role.TOOL_RESPONSE.value),
            )
            for response in responses
        ]

    def __init__(
        self,
        model_name: Optional[str] = None,
        idearium: Idearium = None,
        tools: List[Tool] = None,
        **kwargs,
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
