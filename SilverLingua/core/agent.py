from abc import ABC, abstractmethod
from typing import List, Union

from .atoms import Idearium, Model, Notion, Tool


class Agent(ABC):
    """
    A wrapper around a model that utilizes an Idearium and a set of Tools.
    """

    @property
    @abstractmethod
    def model(self) -> Model:
        """
        The model used by the agent.
        """
        pass

    @property
    @abstractmethod
    def idearium(self) -> Idearium:
        """
        The idearium used by the agent.
        """
        pass

    @property
    @abstractmethod
    def tools(self) -> List[Tool]:
        """
        The tools used by the agent.
        """
        pass

    @abstractmethod
    def add_tool(self, tool: Tool) -> None:
        """
        Adds a tool to the agent.

        Args:
            tool (Tool): The tool to add.
        """
        pass

    @abstractmethod
    def add_tools(self, tools: List[Tool]) -> None:
        """
        Adds a list of tools to the agent.

        Args:
            tools (List[Tool]): The tools to add.
        """
        pass

    @abstractmethod
    def remove_tool(self, tool: Tool) -> None:
        """
        Removes a tool from the agent.

        Args:
            tool (Tool): The tool to remove.
        """
        pass

    @abstractmethod
    def remove_tools(self, tools: List[Tool]) -> None:
        """
        Removes a list of tools from the agent.

        Args:
            tools (List[Tool]): The tools to remove.
        """
        pass

    @abstractmethod
    def generate(
        self, messages: Union[Idearium, List[Idearium]], **kwargs
    ) -> List[Notion]:
        """
        Generates a response to the given messages by calling the
        underlying model's generate method and checking/actualizing tool usage.

        Args:
            messages (Union[Idearium, List[Idearium]]): The messages to respond to.

        Returns:
            List[Notion]: A list of responses to the given messages.
                (Many times there will only be one response.)
        """
        pass

    @abstractmethod
    async def agenerate(
        self, messages: Union[Idearium, List[Idearium]], **kwargs
    ) -> List[Notion]:
        """
        Generates a response to the given messages by calling the
        underlying model's agenerate method and checking/actualizing tool usage.

        Args:
            messages (Union[Idearium, List[Idearium]]): The messages to respond to.

        Returns:
            List[Notion]: A list of responses to the given messages.
                (Many times there will only be one response.)
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> List[Notion]:
        """
        Streams a response to the given prompt by calling the
        underlying model's stream method and checking/actualizing tool usage.

        NOTE: Will raise an exception if the underlying model does not support
        streaming.

        Args:
            prompt (str): The prompt to respond to.

        Returns:
            List[Notion]: A list of responses to the given prompt.
                (Many times there will only be one response.)
        """
        pass
