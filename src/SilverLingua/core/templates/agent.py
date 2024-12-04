import asyncio
import contextlib
import json
import logging
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from ..atoms import ChatRole, Tool, ToolCallResponse, ToolCalls
from ..molecules import Notion
from ..organisms import Idearium
from ..templates.model import Messages, Model

logger = logging.getLogger(__name__)


class Agent(BaseModel):
    """
    A wrapper around a [`Model`][silverlingua.core.templates.model.Model] that utilizes an [`Idearium`][silverlingua.core.organisms.idearium.Idearium] and a set of [`Tool`][silverlingua.core.atoms.tool.tool.Tool]s.

    This is a base class not meant to be used directly. It is meant to be
    subclassed by specific model implementations.

    However, there is limited boilerplate. The only thing that needs to be
    redefined in subclasses is the `_bind_tools` method.

    Additionally, the `_use_tools` method is a common method to redefine.
    """

    model_config = ConfigDict(frozen=True)
    #
    model: Model
    idearium: Idearium
    """
    The Idearium used by the agent.
    """
    tools: List[Tool]
    """
    The tools used by the agent.

    Warning:
        **Do not** modify this list directly. Use `add_tool`, `add_tools`,
        and `remove_tool` instead.
    """
    auto_append_response: bool = True
    """
    Whether to automatically append the response to the idearium after
    generating a response.
    """

    def __init__(
        self,
        model: Model,
        idearium: Optional[Idearium] = None,
        tools: Optional[List[Tool]] = None,
        auto_append_response: bool = True,
    ):
        """
        Initializes the agent.

        Args:
            model (Model): The model to use.
            idearium (Idearium, optional): The idearium to use.
                If None, a new one will be created.
            tools (List[Tool], optional): The tools to use.
        """
        super().__init__(
            model=model,
            idearium=idearium
            or Idearium(tokenizer=model.tokenizer, max_tokens=model.max_tokens),
            tools=tools or [],
            auto_append_response=auto_append_response,
        )

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
                    tool_call=tool_call, response=tool(**tc_function_response)
                )
                responses.append(
                    Notion(
                        content=tc_response.model_dump_json(exclude_none=True),
                        role=str(self.role.TOOL_RESPONSE.value),
                    )
                )
            else:
                responses.append(
                    Notion(
                        content=json.dumps(
                            {
                                "tool_call_id": tool_call.id,
                                "content": "Tool not found",
                                "name": "error",
                            }
                        ),
                        role=str(self.role.TOOL_RESPONSE.value),
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

    def _process_messages(self, messages: Messages) -> List[Notion]:
        """Convert various message types into a list of Notions."""
        if isinstance(messages, str):
            return [Notion(content=messages, role=str(self.role.HUMAN.value))]
        elif isinstance(messages, Notion):
            return [messages]
        elif isinstance(messages, Idearium):
            return messages.notions
        elif isinstance(messages, list):
            return [
                (
                    Notion(content=msg, role=str(self.role.HUMAN.value))
                    if isinstance(msg, str)
                    else msg
                )
                for msg in messages
            ]
        raise ValueError(f"Unsupported message type: {type(messages)}")

    def _process_generation(
        self, responses: List[Notion], is_async=False
    ) -> List[Notion]:
        """Wrapper around shared logic between generate and agenerate."""
        response = responses[0]
        # logger.debug(f"Response: {response}")
        if response.chat_role == ChatRole.TOOL_CALL:
            # logger.debug("Tool call detected")
            # Add the tool call to the idearium
            self.idearium.append(response)
            # Call generate again with the tool response
            tool_calls = ToolCalls.model_validate_json(
                '{"list": ' + response.content + "}"
            )
            tool_response = self._use_tools(tool_calls)
            # logger.debug(f"Tool response: {tool_response}")
            if is_async:
                return self.agenerate(tool_response)
            else:
                return self.generate(tool_response)
        else:
            return responses

    def generate(self, messages: Messages, **kwargs) -> List[Notion]:
        """
        Generates a response to the given messages by calling the
        underlying model's generate method and checking/actualizing tool usage.

        Args:
            messages (Union[str, Notion, Idearium, List[Union[str, Notion]]]):
            The messages to respond to.

        Returns:
            List[Notion]: A list of responses to the given messages.
                (Many times there will only be one response.)
        """
        self.idearium.extend(self._process_messages(messages))
        responses = self.model.generate(self.idearium, **kwargs)
        result = self._process_generation(responses)

        if self.auto_append_response:
            self.idearium.extend(result)

        return result

    async def agenerate(self, messages: Messages, **kwargs) -> List[Notion]:
        """
        Asynchronously generates a response to the given messages by calling the
        underlying model's agenerate method and checking/actualizing tool usage.

        Args:
            messages (Union[str, Notion, Idearium, List[Union[str, Notion]]]):
            The messages to respond to.

        Returns:
            List[Notion]: A list of responses to the given messages.
                (Many times there will only be one response.)
        """
        self.idearium.extend(self._process_messages(messages))
        responses = await self.model.agenerate(self.idearium, **kwargs)
        result = self._process_generation(responses, True)
        r = await result if asyncio.iscoroutine(result) else result

        if self.auto_append_response:
            self.idearium.extend(r)

        return r

    def _process_tool_calls(self, tool_calls: ToolCalls):
        """
        Processes tool calls and returns the tool response.

        Args:
            tool_calls (ToolCalls): The tool calls to process.

        Returns:
            Optional[ToolCalls]: The tool response. If None, no tool calls were found.
        """
        for i, tool_call in enumerate(tool_calls.list):
            if not tool_call.id.startswith("call_"):
                # Something went wrong and this tool call is not valid
                tool_calls.list.pop(i)
                logger.error(
                    "Invalid tool call: "
                    + f"{tool_call.model_dump_json(exclude_none=True)}"
                )

        tc_dump = tool_calls.model_dump(exclude_none=True)
        if tc_dump.get("list"):
            logger.debug(f"Tool calls: {tc_dump}")

            # Create a new notion from the tool calls
            tc_notion = Notion(
                content=json.dumps(tc_dump.get("list")),
                role=str(ChatRole.TOOL_CALL.value),
            )

            # Add the tool call to the idearium
            self.idearium.append(tc_notion)
            # Call stream again with the tool response
            tool_response = self._use_tools(tool_calls)
            return tool_response
        else:
            logger.error("No tool calls found")
            return None

    def stream(self, messages: Messages, **kwargs):
        """
        Streams a response to the given prompt by calling the
        underlying model's stream method and checking/actualizing tool usage.

        NOTE: Will raise an exception if the underlying model does not support
        streaming.

        Args:
            messages (Union[str, Notion, Idearium, List[Union[str, Notion]]]):
            The messages to respond to.

        Returns:
            Generator[Notion, Any, None]: A generator of responses to the given
                messages.
        """
        self.idearium.extend(self._process_messages(messages))
        response_stream = self.model.stream(self.idearium, **kwargs)

        # Process stream directly
        tool_calls: Optional[ToolCalls] = None

        for r in response_stream:
            if r.chat_role == ChatRole.TOOL_CALL:
                logger.debug(f"Tool call detected: {r.content}")
                tc_chunks = ToolCalls.model_validate_json('{"list": ' + r.content + "}")
                tool_calls = tool_calls and tool_calls.concat(tc_chunks) or tc_chunks
                continue
            elif r.content is not None:
                logger.debug(f"Got chunk in stream: {r.content!r}")
                if self.auto_append_response:
                    self.idearium.append(r)
                yield r

        # Handle tool calls if any
        if tool_calls is not None:
            logger.debug("Moving to tool response stream")
            tool_response = self._process_tool_calls(tool_calls)
            if tool_response is not None:
                for r in self.stream(tool_response):
                    yield r

    async def astream(self, messages: Messages, **kwargs):
        """
        Asynchronously streams a response to the given prompt by calling the
        underlying model's astream method and checking/actualizing tool usage.

        NOTE: Will raise an exception if the underlying model does not support
        streaming.

        Args:
            messages (Union[str, Notion, Idearium, List[Union[str, Notion]]]):
            The messages to respond to.

        Returns:
            Generator[Notion, Any, None]: A generator of responses to the given
                messages.
        """
        self.idearium.extend(self._process_messages(messages))
        response_stream = self.model.astream(self.idearium, **kwargs)

        # Process stream directly
        tool_calls: Optional[ToolCalls] = None

        async for r in response_stream:
            if r.chat_role == ChatRole.TOOL_CALL:
                logger.debug(f"Tool call detected: {r.content}")
                tc_chunks = ToolCalls.model_validate_json('{"list": ' + r.content + "}")
                tool_calls = tool_calls and tool_calls.concat(tc_chunks) or tc_chunks
                continue
            elif r.content is not None:
                logger.debug(f"Got chunk in astream: {r.content!r}")
                if self.auto_append_response:
                    self.idearium.append(r)
                yield r

        # Handle tool calls if any
        if tool_calls is not None:
            logger.debug("Moving to tool response stream")
            tool_response = self._process_tool_calls(tool_calls)
            if tool_response is not None:
                async for r in self.astream(tool_response):
                    yield r
