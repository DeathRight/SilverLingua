import json

import pytest

from silverlingua.core.atoms import (
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolCalls,
)
from silverlingua.core.molecules import Notion
from silverlingua.core.templates.agent import Agent
from silverlingua.core.templates.model import ModelType

from .test_model import MockModel


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.model
@pytest.mark.unit
def test_model_first():
    """Test model initialization first."""
    model = MockModel()
    assert model.api_key == "mock-api-key"
    assert model.name == "mock-model"
    assert model.type == ModelType.CHAT
    assert model.can_stream is True
    assert model.max_tokens == 100


def mock_tool_function(x: int) -> int:
    """A simple tool that doubles a number."""
    return x * 2


@pytest.fixture
def mock_tool():
    return Tool(function=mock_tool_function)


@pytest.fixture
def agent(mock_tool):
    model = MockModel()
    return Agent(model=model, tools=[mock_tool])


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.agent
@pytest.mark.unit
def test_agent_initialization(agent, mock_tool):
    """Test basic agent initialization."""
    assert agent.model is not None
    assert len(agent.tools) == 1
    assert agent.tools[0] == mock_tool


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.agent
@pytest.mark.unit
def test_agent_find_tool(agent):
    """Test agent tool finding."""
    tool = agent._find_tool("mock_tool_function")
    assert tool is not None
    assert tool.name == "mock_tool_function"


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.agent
@pytest.mark.unit
def test_agent_use_tools(agent):
    """Test agent tool usage."""
    tool_calls = ToolCalls(
        list=[
            ToolCall(
                function=ToolCallFunction(
                    name="mock_tool_function", arguments=json.dumps({"x": 2})
                )
            )
        ]
    )
    results = agent._use_tools(tool_calls)
    assert len(results) == 1
    assert isinstance(results[0], Notion)
    assert results[0].role == "TOOL_RESPONSE"


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.agent
@pytest.mark.unit
def test_agent_generate(agent):
    """Test agent generation."""
    response = agent.generate("Double the number 2")
    assert isinstance(response, list)
    assert all(isinstance(n, Notion) for n in response)


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.agent
@pytest.mark.unit
async def test_agent_agenerate(agent):
    """Test agent async generation."""
    response = await agent.agenerate("Double the number 2")
    assert isinstance(response, list)
    assert all(isinstance(n, Notion) for n in response)


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.agent
@pytest.mark.unit
def test_agent_stream(agent):
    """Test agent streaming."""
    stream = agent.stream("Double the number 2")
    responses = list(stream)
    assert all(isinstance(n, Notion) for n in responses)


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.agent
@pytest.mark.unit
async def test_agent_astream(agent):
    """Test agent async streaming."""
    stream = agent.astream("Double the number 2")
    responses = [n async for n in stream]
    assert all(isinstance(n, Notion) for n in responses)
