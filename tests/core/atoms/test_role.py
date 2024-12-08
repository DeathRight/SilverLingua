import pytest

from silverlingua.core.atoms.role import (
    ChatRole,
    ReactRole,
    RoleMember,
    create_chat_role,
    create_react_role,
)
from silverlingua.util import ImmutableAttributeError


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.role
@pytest.mark.unit
def test_chat_role_basic():
    """Test basic ChatRole functionality."""
    # Test all standard roles exist
    assert ChatRole.SYSTEM
    assert ChatRole.HUMAN
    assert ChatRole.AI
    assert ChatRole.TOOL_CALL
    assert ChatRole.TOOL_RESPONSE

    # Test role values
    assert str(ChatRole.SYSTEM.value) == "SYSTEM"
    assert str(ChatRole.HUMAN.value) == "HUMAN"
    assert str(ChatRole.AI.value) == "AI"
    assert str(ChatRole.TOOL_CALL.value) == "TOOL_CALL"
    assert str(ChatRole.TOOL_RESPONSE.value) == "TOOL_RESPONSE"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.role
@pytest.mark.unit
def test_react_role_basic():
    """Test basic ReactRole functionality."""
    # Test all standard roles exist
    assert ReactRole.THOUGHT
    assert ReactRole.OBSERVATION
    assert ReactRole.ACTION
    assert ReactRole.QUESTION
    assert ReactRole.ANSWER

    # Test role values
    assert str(ReactRole.THOUGHT.value) == "THOUGHT"
    assert str(ReactRole.OBSERVATION.value) == "OBSERVATION"
    assert str(ReactRole.ACTION.value) == "ACTION"
    assert str(ReactRole.QUESTION.value) == "QUESTION"
    assert str(ReactRole.ANSWER.value) == "ANSWER"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.role
@pytest.mark.unit
def test_role_member():
    """Test RoleMember functionality."""
    # Create a role member
    member = RoleMember("TEST", "test_value")

    # Test properties
    assert member.name == "TEST"
    assert member.value == "test_value"
    assert str(member) == "test_value"

    # Test immutability
    with pytest.raises(ImmutableAttributeError):
        member.name = "NEW_TEST"

    with pytest.raises(ImmutableAttributeError):
        member.value = "new_value"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.role
@pytest.mark.unit
def test_role_member_equality():
    """Test RoleMember equality comparison."""
    # Create role members with same parent
    parent = object()
    member1 = RoleMember("TEST", "test1", parent)
    member2 = RoleMember("TEST", "test2", parent)  # Different value, same name
    member3 = RoleMember("OTHER", "test1", parent)  # Different name
    member4 = RoleMember("TEST", "test1", object())  # Different parent

    # Test equality
    assert member1 == member2  # Same name and parent
    assert member1 != member3  # Different name
    assert member1 != member4  # Different parent
    assert member1 != "TEST"  # Different type


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.role
@pytest.mark.unit
def test_create_chat_role():
    """Test chat role creation."""
    # Create a custom chat role
    CustomRole = create_chat_role(
        "CustomRole",
        SYSTEM="sys",
        HUMAN="user",
        AI="bot",
        TOOL_CALL="call",
        TOOL_RESPONSE="response",
    )

    # Test role values
    assert str(CustomRole.SYSTEM.value) == "sys"
    assert str(CustomRole.HUMAN.value) == "user"
    assert str(CustomRole.AI.value) == "bot"
    assert str(CustomRole.TOOL_CALL.value) == "call"
    assert str(CustomRole.TOOL_RESPONSE.value) == "response"

    # Test equality with base ChatRole
    assert CustomRole.SYSTEM == ChatRole.SYSTEM
    assert CustomRole.HUMAN == ChatRole.HUMAN
    assert CustomRole.AI == ChatRole.AI
    assert CustomRole.TOOL_CALL == ChatRole.TOOL_CALL
    assert CustomRole.TOOL_RESPONSE == ChatRole.TOOL_RESPONSE


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.role
@pytest.mark.unit
def test_create_react_role():
    """Test react role creation."""
    # Create a custom react role
    CustomReact = create_react_role(
        "CustomReact",
        THOUGHT="thinking",
        OBSERVATION="observing",
        ACTION="acting",
        QUESTION="asking",
        ANSWER="answering",
    )

    # Test role values
    assert str(CustomReact.THOUGHT.value) == "thinking"
    assert str(CustomReact.OBSERVATION.value) == "observing"
    assert str(CustomReact.ACTION.value) == "acting"
    assert str(CustomReact.QUESTION.value) == "asking"
    assert str(CustomReact.ANSWER.value) == "answering"

    # Test equality with base ReactRole
    assert CustomReact.THOUGHT == ReactRole.THOUGHT
    assert CustomReact.OBSERVATION == ReactRole.OBSERVATION
    assert CustomReact.ACTION == ReactRole.ACTION
    assert CustomReact.QUESTION == ReactRole.QUESTION
    assert CustomReact.ANSWER == ReactRole.ANSWER


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.role
@pytest.mark.unit
def test_role_member_parent_setting():
    """Test that role member parent can only be set once."""
    member = RoleMember("TEST", "test")

    # First parent setting should work
    member._parent = ChatRole
    assert member._parent == ChatRole

    # Second parent setting should fail
    with pytest.raises(ImmutableAttributeError):
        member._parent = ReactRole
