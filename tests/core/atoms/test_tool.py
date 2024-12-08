import json
from typing import Optional

import pytest

from silverlingua.core.atoms.tool import (
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolCalls,
    tool,
)


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_basic_tool_creation():
    """Test basic tool creation using the Tool class."""

    def sample_function(x: int) -> int:
        """Multiply a number by 2."""
        return x * 2

    tool_instance = Tool(function=sample_function)
    assert tool_instance.function == sample_function
    assert tool_instance.name == "sample_function"
    assert "Multiply a number by 2" in tool_instance.description.description


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_execution():
    """Test tool execution with direct arguments."""

    def sample_function(x: int) -> int:
        return x * 2

    tool_instance = Tool(function=sample_function)
    result = tool_instance(2)
    assert result == json.dumps(4)


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_call_function():
    """Test tool execution with ToolCallFunction."""

    def sample_function(x: int) -> int:
        return x * 2

    tool_instance = Tool(function=sample_function)
    function_call = ToolCallFunction(
        name="sample_function", arguments=json.dumps({"x": 2})
    )
    result = tool_instance(function_call)
    assert result == json.dumps(4)


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_decorator_basic():
    """Test basic tool decorator functionality."""

    @tool
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y

    # Test function execution
    result = add_numbers(2, 3)
    assert result == json.dumps(5)

    # Test tool attributes
    assert add_numbers.name == "add_numbers"
    assert "Add two numbers together" in add_numbers.description.description


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_decorator_with_optional():
    """Test tool decorator with optional parameters."""

    @tool
    def greet(name: str, title: Optional[str] = None) -> str:
        """Greet someone with an optional title."""
        if title:
            return f"Hello, {title} {name}!"
        return f"Hello, {name}!"

    # Test without optional parameter
    result = greet("Alice")
    assert result == json.dumps("Hello, Alice!")

    # Test with optional parameter
    result = greet("Alice", "Ms.")
    assert result == json.dumps("Hello, Ms. Alice!")


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_decorator_with_tool_call():
    """Test tool decorator with ToolCallFunction."""

    @tool
    def multiply(x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y

    function_call = ToolCallFunction(
        name="multiply", arguments=json.dumps({"x": 2.5, "y": 3.0})
    )
    result = multiply(function_call)
    assert result == json.dumps(7.5)


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_error_handling():
    """Test error handling in tools."""

    @tool
    def divide(x: float, y: float) -> float:
        """Divide two numbers."""
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y

    # Test normal case
    result = divide(6.0, 2.0)
    assert result == json.dumps(3.0)

    # Test error case
    with pytest.raises(ValueError):
        divide(1.0, 0.0)


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_docstring_parsing():
    """Test that tool properly parses function docstrings."""

    @tool
    def complex_function(name: str, age: int, scores: list[float]) -> dict:
        """
        Process user data and calculate statistics.

        Args:
            name: The user's name
            age: The user's age
            scores: List of test scores

        Returns:
            dict: User stats including average score
        """
        return {"name": name, "age": age, "average_score": sum(scores) / len(scores)}

    # Check that docstring was properly parsed into description
    assert "Process user data" in complex_function.description.description

    # Test function execution
    result = complex_function("Alice", 25, [85.0, 90.0, 95.0])
    expected = {"name": "Alice", "age": 25, "average_score": 90.0}
    assert json.loads(result) == expected


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_json_serialization():
    """Test that tool can be properly serialized to JSON."""

    @tool
    def sample_function(x: int, y: str, flag: bool = False) -> dict:
        """A sample function with multiple parameter types."""
        return {"x": x, "y": y, "flag": flag}

    # Convert tool to string (JSON)
    tool_json = str(sample_function)

    # Parse the JSON and verify structure
    tool_dict = json.loads(tool_json)
    assert "name" in tool_dict
    assert "description" in tool_dict
    assert "parameters" in tool_dict["description"]

    # Verify parameter types
    params = tool_dict["description"]["parameters"]["properties"]
    assert params["x"]["type"] == "integer"
    assert params["y"]["type"] == "string"
    assert params["flag"]["type"] == "boolean"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_invalid_arguments():
    """Test tool behavior with invalid arguments."""

    @tool
    def process_data(data: list[int]) -> int:
        """Process a list of integers."""
        return sum(data)

    # Test with invalid JSON
    invalid_call = ToolCallFunction(name="process_data", arguments="invalid json")
    with pytest.raises(ValueError) as exc_info:
        Tool(function=process_data)(invalid_call)
    assert "must be a JSON string" in str(exc_info.value)

    # Test with wrong argument type
    wrong_type_call = ToolCallFunction(
        name="process_data", arguments=json.dumps({"data": ["not", "integers"]})
    )
    with pytest.raises(TypeError):
        Tool(function=process_data)(wrong_type_call)


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_calls_json_validation():
    """Test ToolCalls JSON validation and parsing."""
    # Valid JSON format
    valid_json = (
        '{"list": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]}'
    )
    tool_calls = ToolCalls.model_validate_json(valid_json)
    assert len(tool_calls.list) == 1
    assert tool_calls.list[0].id == "call_1"
    assert tool_calls.list[0].function.name == "test"

    # Invalid JSON format - missing required 'function' field
    with pytest.raises(Exception):  # Pydantic raises its own validation error
        ToolCalls.model_validate_json('{"list": [{"id": "call_1"}]}')

    # Invalid JSON syntax
    with pytest.raises(Exception):
        ToolCalls.model_validate_json('{"list": [not valid json]}')


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_calls_concatenation():
    """Test ToolCalls concatenation behavior."""
    # Create two tool calls with same ID but different arguments
    call1 = ToolCall(
        id="call_1",
        function=ToolCallFunction(name="test", arguments=json.dumps({"x": 1})),
    )
    call2 = ToolCall(
        id="call_1",  # Same ID
        function=ToolCallFunction(name="test", arguments=json.dumps({"y": 2})),
    )

    # Create ToolCalls objects
    calls1 = ToolCalls(list=[call1])
    calls2 = ToolCalls(list=[call2])

    # Test concatenation
    merged = calls1.concat(calls2)
    assert len(merged.list) == 1  # Should merge into one call
    merged_call = merged.list[0]
    assert merged_call.id == "call_1"

    # For now, we expect concatenated arguments to be separate JSON objects
    # This matches the current implementation's behavior
    assert merged_call.function.arguments == json.dumps({"x": 1}) + json.dumps({"y": 2})


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_calls_streaming():
    """Test ToolCalls handling of streaming responses."""
    # Test case 1: Complete function name in first chunk, arguments split across chunks
    part1 = ToolCalls.model_validate_json(
        '{"list": [{"id": "call_1", "function": {"name": "test", "arguments": ""}}]}'
    )
    part2 = ToolCalls.model_validate_json(
        '{"list": [{"id": "call_1", "function": {"name": "", "arguments": "{\\"x\\": 1}"}}]}'
    )

    # Merge the parts
    merged = part1.concat(part2)
    assert len(merged.list) == 1

    # Verify merged result
    merged_call = merged.list[0]
    assert merged_call.id == "call_1"
    assert merged_call.function.name == "test"  # Name should come from first chunk
    assert (
        merged_call.function.arguments == '{"x": 1}'
    )  # Arguments should be complete JSON

    # Test case 2: Arguments split at token boundary
    part1 = ToolCalls.model_validate_json(
        '{"list": [{"id": "call_2", "function": {"name": "test", "arguments": "{\\"message\\": \\"Hello"}}]}'
    )
    part2 = ToolCalls.model_validate_json(
        '{"list": [{"id": "call_2", "function": {"name": "", "arguments": " World\\"}"}}]}'
    )

    # Merge the parts
    merged = part1.concat(part2)
    assert len(merged.list) == 1

    # Verify merged result
    merged_call = merged.list[0]
    assert merged_call.id == "call_2"
    assert merged_call.function.name == "test"
    assert (
        merged_call.function.arguments == '{"message": "Hello World"}'
    )  # Should handle string concatenation


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.tool
@pytest.mark.unit
def test_tool_calls_ordering():
    """Test that ToolCalls maintains order and handles indices."""
    # Create tool calls with indices
    call1 = ToolCall(id="call_1", index=0, function=ToolCallFunction(name="test1"))
    call2 = ToolCall(id="call_2", index=1, function=ToolCallFunction(name="test2"))
    call3 = ToolCall(id="call_3", index=2, function=ToolCallFunction(name="test3"))

    # Create ToolCalls in different orders
    calls1 = ToolCalls(list=[call1, call2])
    calls2 = ToolCalls(list=[call3])

    # Test concatenation preserves order
    merged = calls1.concat(calls2)
    assert len(merged.list) == 3
    assert [call.index for call in merged.list] == [0, 1, 2]
    assert [call.function.name for call in merged.list] == ["test1", "test2", "test3"]
