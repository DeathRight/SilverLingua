"""Tests for the Memory atom."""

import json

import pytest

from silverlingua.core.atoms import Memory


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.memory
@pytest.mark.unit
def test_memory_initialization():
    """Test that Memory can be initialized with different content types."""
    # Empty memory
    empty_memory = Memory(content="")
    assert empty_memory.content == ""

    # Memory with string content
    memory = Memory(content="test content")
    assert memory.content == "test content"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.memory
@pytest.mark.unit
def test_memory_json_storage():
    """Test storing and retrieving JSON-serialized data in memory."""
    memory = Memory(content="")

    # Store and retrieve single value
    data = {"test_key": "test_value"}
    memory.content = json.dumps(data)
    loaded_data = json.loads(memory.content)
    assert loaded_data["test_key"] == "test_value"

    # Store and retrieve multiple values
    test_data = {
        "string": "value",
        "number": 42,
        "list": [1, 2, 3],
        "dict": {"nested": "data"},
    }

    memory.content = json.dumps(test_data)
    loaded_data = json.loads(memory.content)
    assert loaded_data == test_data


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.memory
@pytest.mark.unit
def test_memory_update():
    """Test updating memory content."""
    memory = Memory(content="old content")

    # Update with new string
    memory.content = "new content"
    assert memory.content == "new content"

    # Update with JSON string
    data = {"key": "value"}
    memory.content = json.dumps(data)
    assert json.loads(memory.content)["key"] == "value"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.memory
@pytest.mark.unit
def test_memory_str():
    """Test string representation of Memory."""
    # Empty memory
    empty_memory = Memory(content="")
    assert str(empty_memory) == ""

    # Memory with simple content
    simple_memory = Memory(content="test content")
    assert str(simple_memory) == "test content"

    # Memory with JSON content
    data = {
        "string": "value",
        "number": 42,
        "list": [1, 2, 3],
        "dict": {"nested": "data"},
    }
    json_memory = Memory(content=json.dumps(data))
    loaded_str = json.loads(str(json_memory))
    assert loaded_str == data
