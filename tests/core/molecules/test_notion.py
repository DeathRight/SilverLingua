import pytest

from SilverLingua.core.atoms import ChatRole, ReactRole
from SilverLingua.core.molecules import Notion


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_notion_initialization():
    """Test basic notion initialization."""
    # Test with string role
    notion = Notion(content="Hello", role="SYSTEM")
    assert notion.content == "Hello"
    assert notion.role == "SYSTEM"
    assert not notion.persistent

    # Test with ChatRole
    notion = Notion(content="Hello", role=ChatRole.SYSTEM)
    assert notion.content == "Hello"
    assert notion.role == str(ChatRole.SYSTEM.value)
    assert not notion.persistent

    # Test with ReactRole
    notion = Notion(content="Hello", role=ReactRole.THOUGHT)
    assert notion.content == "Hello"
    assert notion.role == str(ReactRole.THOUGHT.value)
    assert not notion.persistent

    # Test with persistence
    notion = Notion(content="Hello", role="SYSTEM", persistent=True)
    assert notion.persistent


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_notion_str():
    """Test string representation."""
    notion = Notion(content="Hello", role="SYSTEM")
    assert str(notion) == "SYSTEM: Hello"


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_notion_chat_role():
    """Test chat role conversion."""
    # Test with valid chat role
    notion = Notion(content="Hello", role=ChatRole.SYSTEM)
    assert notion.chat_role == ChatRole.SYSTEM

    # Test with string chat role
    notion = Notion(content="Hello", role="SYSTEM")
    assert notion.chat_role == ChatRole.SYSTEM

    # Test with unknown role (should default to AI)
    notion = Notion(content="Hello", role="UNKNOWN")
    assert notion.chat_role == ChatRole.AI


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_notion_react_role():
    """Test react role conversion."""
    # Test with valid react role
    notion = Notion(content="Hello", role=ReactRole.THOUGHT)
    assert notion.react_role == ReactRole.THOUGHT

    # Test with string react role
    notion = Notion(content="Hello", role="THOUGHT")
    assert notion.react_role == ReactRole.THOUGHT

    # Test with unknown role (should default to THOUGHT)
    notion = Notion(content="Hello", role="UNKNOWN")
    assert notion.react_role == ReactRole.THOUGHT


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_notion_validation():
    """Test role validation."""
    # Valid roles should work
    Notion(content="Hello", role=ChatRole.SYSTEM)
    Notion(content="Hello", role=ReactRole.THOUGHT)
    Notion(content="Hello", role="SYSTEM")

    # Invalid role type should fail
    with pytest.raises(ValueError):
        Notion(content="Hello", role=123)  # type: ignore
