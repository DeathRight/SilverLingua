import json
from typing import List, Union

from ..atoms import ChatRole, Idearium, Model, Notion, Tool
from ..atoms.tool import FunctionCall, FunctionResponse


class Agent:
    """
    A wrapper around a model that utilizes an Idearium and a set of Tools.

    This is a base class not meant to be used directly. It is meant to be
    subclassed by specific model implementations.

    However, there is limited boilerplate. The only thing that needs to be
    redefined in subclasses is the _bind_tools method.
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
    def role(self) -> ChatRole:
        """
        The ChatRole object for the model.
        """
        return self._model.role

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

        Do not modify this list directly. Use `add_tool`, `add_tools`,
        and `remove_tool` instead.
        """
        return self._tools

    def _find_tool(self, name: str) -> Tool | None:
        """
        Finds a tool by name.
        """
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def _use_tool(self, notion: Notion) -> List[Notion] | Notion:
        """
        Uses a tool or tools based on the given notion, turning the notion's
        content into a FunctionCall or FunctionCall list and calling the
        appropriate tool(s).

        Args:
            notion (Notion): The notion that's content contains the
            FunctionCall(s).

        Returns:
            List[Notion] | Notion: The response(s) to the tool call(s). The role
            will be ChatRole.TOOL_RESPONSE.
        """
        content = json.loads(notion.content)
        if isinstance(content, list):
            responses: List[Notion] = []
            for tc in content:
                fc = FunctionCall.from_json(tc)
                tool = self._find_tool(fc.name)
                if tool is not None:
                    responses.append(
                        Notion(
                            FunctionResponse(fc.name, tool(fc)).to_json(),
                            str(self.role.TOOL_RESPONSE.value),
                        )
                    )
                else:
                    responses.append(
                        Notion(
                            FunctionResponse("error", "Tool not found").to_json(),
                            str(self.role.TOOL_RESPONSE.value),
                        )
                    )
            return responses
        else:
            fc = FunctionCall.from_json(content)
            tool = self._find_tool(fc.name)
            if tool is not None:
                return Notion(
                    FunctionResponse(fc.name, tool(fc)).to_json(),
                    str(self.role.TOOL_RESPONSE.value),
                )
            else:
                return Notion(
                    FunctionResponse("error", "Tool not found").to_json(),
                    str(self.role.TOOL_RESPONSE.value),
                )

    def _bind_tools(self) -> None:
        """
        Called at the end of __init__ to bind the tools to the model.

        This MUST be redefined in subclasses to dictate how
        the tools are bound to the model.
        """
        pass

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

        self._bind_tools()

    def add_tool(self, tool: Tool) -> None:
        """
        Adds a tool to the agent.
        """
        self._tools.append(tool)
        self._bind_tools()

    def add_tools(self, tools: List[Tool]) -> None:
        """
        Adds a list of tools to the agent.
        """
        self._tools.extend(tools)
        self._bind_tools()

    def remove_tool(self, name: str) -> None:
        """
        Removes a tool from the agent.
        """
        for i, tool in enumerate(self._tools):
            if tool.name == name:
                self._tools.pop(i)
                break
        self._bind_tools()

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
        responses = self._model.generate(self._idearium, **kwargs)

        response = responses[0]
        if response.chat_role == ChatRole.TOOL_CALL:
            # Call generate again with the tool response
            tool_response = self._use_tool(response)
            if isinstance(tool_response, list):
                return self.generate(tool_response)
            else:
                return self.generate([self._use_tool(response)])
        else:
            return responses

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
        self._idearium.extend(messages)
        responses = await self._model.agenerate(self._idearium, **kwargs)

        response = responses[0]
        if response.chat_role == ChatRole.TOOL_CALL:
            # Call agenerate again with the tool response
            tool_response = self._use_tool(response)
            if isinstance(tool_response, list):
                return await self.agenerate(tool_response)
            else:
                return await self.agenerate([tool_response])
        else:
            return responses

    def stream(self, messages: Union[Idearium, List[Notion]], **kwargs) -> Notion:
        """
        Streams a response to the given prompt by calling the
        underlying model's stream method and checking/actualizing tool usage.

        NOTE: Will raise an exception if the underlying model does not support
        streaming.

        Args:
            messages (Union[Idearium, List[Notion]]): The messages to respond to.

        Returns:
            Notion: The model's response, one token at a time.
        """
        pass
