import contextlib
import json
import logging
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..atoms import ChatRole, Idearium, Model, Notion, Tool
from ..atoms.tool import (
    ToolCallResponse,
    ToolCalls,
)

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """
    A wrapper around a model that utilizes an Idearium and a set of Tools.

    This is a base class not meant to be used directly. It is meant to be
    subclassed by specific model implementations.

    However, there is limited boilerplate. The only thing that needs to be
    redefined in subclasses is the `_bind_tools` method.

    Additionally, the `_use_tools` method is a common method to redefine.
    """

    model_config = ConfigDict(frozen=True)
    #
    model: Model
    idearium: Idearium = Field(validate_default=True)
    """
    The Idearium used by the agent.

    WARNING: Do not modify this list directly.
    """
    tools: List[Tool] = Field(default_factory=list)
    """
    The tools used by the agent.

    WARNING: Do not modify this list directly. Use `add_tool`, `add_tools`,
    and `remove_tool` instead.
    """

    @field_validator("idearium", mode="before")
    def set_default_idearium(cls, v, values):
        if v is None:
            model = values.get("model")
            if model is not None:
                return Idearium(tokenizer=model.tokenizer, max_tokens=model.max_tokens)
        return v

    def model_post_init(self, __content):
        self._bind_tools()

    @property
    def model(self) -> Model:
        """
        The model used by the agent.
        """
        return self.model

    @property
    def role(self) -> ChatRole:
        """
        The ChatRole object for the model.
        """
        return self.model.role

    def _find_tool(self, name: str) -> Tool | None:
        """
        Finds a tool by name.
        """
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def _use_tools(self, tool_calls: ToolCalls) -> List[Notion]:
        """
        Uses Tools based on the given ToolCalls, returning Notions
        containing ToolCallResponses.

        Args:
            tool_calls (ToolCalls): The ToolCalls to use.

        Returns:
            List[Notion]: The Notions containing ToolCallResponses.
                Each Notion will have a role of ChatRole.TOOL_RESPONSE.
        """
        responses: List[Notion] = []
        for tool_call in tool_calls.list:
            tool = self._find_tool(tool_call.function.name)
            if tool is not None:
                tc_function_response = {}
                with contextlib.suppress(json.JSONDecodeError):
                    tc_function_response = json.loads(tool_call.function.arguments)

                tc_response = ToolCallResponse.from_tool_call(
                    tool_call, tool(**tc_function_response)
                )
                responses.append(
                    Notion(
                        tc_response.model_dump_json(),
                        str(self.role.TOOL_RESPONSE.value),
                    )
                )
            else:
                responses.append(
                    Notion(
                        json.dumps(
                            {
                                "tool_call_id": tool_call.id,
                                "content": "Tool not found",
                                "name": "error",
                            }
                        ),
                        str(self.role.TOOL_RESPONSE.value),
                    )
                )
        return responses

    def _bind_tools(self) -> None:
        """
        Called at the end of __init__ to bind the tools to the model.

        This MUST be redefined in subclasses to dictate how
        the tools are bound to the model.

        Example:
        ```python
        # From OpenAIChatAgent
        def _bind_tools(self) -> None:
            m_tools: List[ChatCompletionToolParam] = [
                {"type": "function", "function": tool.description}
                for tool in self.tools
            ]

            if len(m_tools) > 0:
                self.model.tools = m_tools
        ```
        """
        pass

    def add_tool(self, tool: Tool) -> None:
        """
        Adds a tool to the agent.
        """
        self.tools.append(tool)
        self._bind_tools()

    def add_tools(self, tools: List[Tool]) -> None:
        """
        Adds a list of tools to the agent.
        """
        self.tools.extend(tools)
        self._bind_tools()

    def remove_tool(self, name: str) -> None:
        """
        Removes a tool from the agent.
        """
        for i, tool in enumerate(self.tools):
            if tool.name == name:
                self.tools.pop(i)
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
        self.idearium.extend(messages)
        responses = self.model.generate(self.idearium, **kwargs)

        response = responses[0]
        # logger.debug(f"Response: {response}")
        if response.chat_role == ChatRole.TOOL_CALL:
            # logger.debug("Tool call detected")
            # Add the tool call to the idearium
            self.idearium.append(response)
            # Call generate again with the tool response
            tool_response = self._use_tools(response)
            # logger.debug(f"Tool response: {tool_response}")
            return self.generate(tool_response)
        else:
            return responses

    async def agenerate(
        self, messages: Union[Idearium, List[Notion]], **kwargs
    ) -> List[Notion]:
        """
        Asynchronously generates a response to the given messages by calling the
        underlying model's agenerate method and checking/actualizing tool usage.

        Args:
            messages (Union[Idearium, List[Notion]]): The messages to respond to.

        Returns:
            List[Notion]: A list of responses to the given messages.
                (Many times there will only be one response.)
        """
        self.idearium.extend(messages)
        responses = await self.model.agenerate(self.idearium, **kwargs)

        response = responses[0]
        # logger.debug(f"Response: {response}")
        if response.chat_role == ChatRole.TOOL_CALL:
            # logger.debug("Tool call detected")
            # Add the tool call to the idearium
            self.idearium.append(response)
            # Call generate again with the tool response
            tool_response = self._use_tools(response)
            # logger.debug(f"Tool response: {tool_response}")
            return await self.agenerate(tool_response)
        else:
            return responses

    def stream(self, messages: Union[Idearium, List[Notion]], **kwargs):
        """
        Streams a response to the given prompt by calling the
        underlying model's stream method and checking/actualizing tool usage.

        NOTE: Will raise an exception if the underlying model does not support
        streaming.

        Args:
            messages (Union[Idearium, List[Notion]]): The messages to respond to.

        Returns:
            Generator[Notion, Any, None]: A generator of responses to the given
                messages.
        """
        self.idearium.extend(messages)
        response = self.model.stream(self.idearium, **kwargs)

        tool_calls: Optional[ToolCalls] = None
        for r in response:
            if r.chat_role == ChatRole.TOOL_CALL:
                logger.debug(f"Tool call detected: {r.content}")

                tc_chunks = ToolCalls.model_validate_json(
                    '{"tool_calls": ' + r.content + "}"
                )
                tool_calls = tool_calls and tool_calls.concat(tc_chunks) or tc_chunks
                continue
            elif r.content is not None:
                logger.debug(f"Response: {r}")
                yield r
            continue

        if tool_calls is not None:
            logger.debug("Moving to tool response stream")

            for i, tool_call in enumerate(tool_calls.list):
                if not tool_call.id.startswith("call_"):
                    # Something went wrong and this tool call is not valid
                    tool_calls.list.pop(i)
                    logger.error(f"Invalid tool call: {tool_call.model_dump_json()}")

            tc_notion = Notion(
                content=json.dumps(tool_calls.model_dump().list),
                chat_role=str(ChatRole.TOOL_CALL.value),
            )

            # Add the tool call to the idearium
            self.idearium.append(tc_notion)
            # Call stream again with the tool response
            tool_response = self._use_tools(tc_notion)

            tool_response_stream = self.stream(tool_response)

            if tool_response_stream is not None:
                # Recursively yield the tool response stream
                for r in tool_response_stream:
                    yield r

    async def astream(self, messages: Union[Idearium, List[Notion]], **kwargs):
        """
        Asynchronously streams a response to the given prompt by calling the
        underlying model's astream method and checking/actualizing tool usage.

        NOTE: Will raise an exception if the underlying model does not support
        streaming.

        Args:
            messages (Union[Idearium, List[Notion]]): The messages to respond to.

        Returns:
            Generator[Notion, Any, None]: A generator of responses to the given
                messages.
        """
        self.idearium.extend(messages)
        response = await self.model.astream(self.idearium, **kwargs)

        tool_calls: Optional[ToolCalls] = None
        for r in response:
            if r.chat_role == ChatRole.TOOL_CALL:
                logger.debug(f"Tool call detected: {r.content}")

                tc_chunks = ToolCalls.model_validate_json(
                    '{"tool_calls": ' + r.content + "}"
                )
                tool_calls = tool_calls and tool_calls.concat(tc_chunks) or tc_chunks
                continue
            elif r.content is not None:
                logger.debug(f"Response: {r}")
                yield r
            continue

        if tool_calls is not None:
            logger.debug("Moving to tool response stream")

            for i, tool_call in enumerate(tool_calls.list):
                if not tool_call.id.startswith("call_"):
                    # Something went wrong and this tool call is not valid
                    tool_calls.list.pop(i)
                    logger.error(f"Invalid tool call: {tool_call.model_dump_json()}")

            tc_notion = Notion(
                content=json.dumps(tool_calls.model_dump().list),
                chat_role=str(ChatRole.TOOL_CALL.value),
            )

            # Add the tool call to the idearium
            self.idearium.append(tc_notion)
            # Call stream again with the tool response
            tool_response = self._use_tools(tc_notion)

            tool_response_stream = self.stream(tool_response)

            if tool_response_stream is not None:
                # Recursively yield the tool response stream
                for r in tool_response_stream:
                    yield r
