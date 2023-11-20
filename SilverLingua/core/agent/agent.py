from typing import List, Union

from ..atoms import ChatRole, Idearium, Model, Notion, Tool
from ..atoms.tool import FunctionCall, FunctionResponse


class Agent:
    """
    A wrapper around a model that utilizes an Idearium and a set of Tools.
    """

    _tools: List[Tool]
    _idearium: Idearium
    _model: Model

    @property
    def model(self) -> Model:
        """
        The model used by the agent.
        """
        return self._model

    @property
    def idearium(self) -> Idearium:
        """
        The idearium used by the agent.
        """
        return self._idearium

    @property
    def tools(self) -> List[Tool]:
        """
        The tools used by the agent.
        """
        return self._tools

    def __init__(
        self, model: Model, idearium: Idearium = None, tools: List[Tool] = None
    ) -> None:
        """
        Initializes the agent.
        """
        self._model = model
        self._idearium = (
            idearium
            if idearium is not None
            else Idearium(self._model.tokenizer, self._model.max_tokens)
        )
        self._tools = tools if tools is not None else []

    def _find_tool(self, name: str) -> Tool | None:
        """
        Finds a tool by name.
        """
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def _use_tool(self, notion: Notion) -> Notion:
        """
        Uses a tool based on a Notion by converting it to a FunctionCall.

        Returns a Notion with the response from the tool as the content and
        the role as TOOL_RESPONSE.
        """
        response: FunctionResponse

        fc = FunctionCall.from_json(notion.content)
        tool = self._find_tool(fc.name)
        if tool is not None:
            response = FunctionResponse(fc.name, tool(fc))
        else:
            response = FunctionResponse("error", "Tool not found")

        return Notion(response.to_json(), ChatRole.TOOL_RESPONSE)

    def generate(
        self, messages: Union[Idearium, List[Notion]], **kwargs
    ) -> List[Notion]:
        """
        Generates a response to the given messages by calling the
        underlying model's generate method and checking/actualizing tool usage.

        Args:
            messages (Union[Idearium, List[Notion]]): The messages to respond to.

        Returns:
            List[Notion]: A list of responses to the given messages.
                (Many times there will only be one response.)
        """
        self._idearium.extend(messages)
        response = self._model.generate(self._idearium, tools=self.tools, **kwargs)[0]

        if response.chat_role == ChatRole.TOOL_CALL:
            # Call generate again with the tool response
            return self.generate([self._use_tool(response)])

    async def agenerate(
        self, messages: Union[Idearium, List[Notion]], **kwargs
    ) -> List[Notion]:
        """
        Generates a response to the given messages by calling the
        underlying model's agenerate method and checking/actualizing tool usage.

        Args:
            messages (Union[Idearium, List[Notion]]): The messages to respond to.

        Returns:
            List[Notion]: A list of responses to the given messages.
                (Many times there will only be one response.)
        """
        pass

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
