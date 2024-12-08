import logging
from typing import List, Optional

from silverlingua.core.atoms import Tool
from silverlingua.core.organisms import Idearium
from silverlingua.core.templates.agent import Agent

from ..model import AnthropicModel
from ..model.util import AnthropicModelName, CompletionParams

logger = logging.getLogger(__name__)


class AnthropicChatAgent(Agent):
    """
    An agent that uses the Anthropic chat completion API.
    """

    model: AnthropicModel

    @property
    def model(self) -> AnthropicModel:
        return self._model

    def _bind_tools(self) -> None:
        """Bind tools to the model."""
        if not self.tools:
            return

        tools = []
        for tool in self.tools:
            tools.append(
                {
                    "name": tool.description.name,
                    "description": tool.description.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            name: {
                                "type": param.type,
                                "description": param.description or "",
                            }
                            for name, param in tool.description.parameters.properties.items()
                        },
                        "required": tool.description.parameters.required or [],
                    },
                }
            )

        if tools:
            self.model.completion_params.tools = tools

    def __init__(
        self,
        model_name: AnthropicModelName = "claude-3-opus-20240229",
        idearium: Optional[Idearium] = None,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None,
        completion_params: Optional[CompletionParams] = None,
    ):
        """
        Initialize the Anthropic chat agent.

        Args:
            model_name: The name of the Anthropic model to use
            idearium: The idearium to use. If None, a new one will be created
            tools: The tools to use. If None, no tools will be used
            api_key: The Anthropic API key to use. If None, will attempt to use os.getenv("ANTHROPIC_API_KEY")
            completion_params: The completion parameters to use
        """
        model = AnthropicModel(
            name=model_name,
            api_key=api_key,
            completion_params=completion_params,
        )

        args = {"model": model}
        if idearium is not None:
            args["idearium"] = idearium
        if tools is not None:
            args["tools"] = tools

        super().__init__(**args)
