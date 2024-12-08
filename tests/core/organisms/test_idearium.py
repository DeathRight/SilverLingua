import pytest

from silverlingua.core.atoms import ChatRole, Tokenizer
from silverlingua.core.molecules import Notion
from silverlingua.core.organisms import Idearium


class MockTokenizer(Tokenizer):
    """Mock tokenizer that counts characters as tokens."""

    def __init__(self):
        text_cache = {}

        def encode(text: str) -> list[int]:
            tokens = list(range(len(text)))  # Each character is a token
            # Store the text for later decoding
            for i, token in enumerate(tokens):
                text_cache[token] = text[i]
            return tokens

        def decode(tokens: list[int]) -> str:
            # Reconstruct text from cached characters
            return "".join(text_cache.get(token, "x") for token in tokens)

        super().__init__(encode=encode, decode=decode)


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.idearium
@pytest.mark.unit
def test_idearium_initialization(tokenizer):
    """Test basic Idearium initialization."""
    # Empty initialization
    idearium = Idearium(tokenizer=tokenizer, max_tokens=10)
    assert len(idearium) == 0
    assert idearium.total_tokens == 0

    # Initialize with notions
    notions = [
        Notion(content="Hello", role=ChatRole.HUMAN),
        Notion(content="Hi", role=ChatRole.AI),
    ]
    idearium = Idearium(tokenizer=tokenizer, max_tokens=10, notions=notions)
    assert len(idearium) == 2
    assert idearium.total_tokens == 7  # "Hello" (5) + "Hi" (2)


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.idearium
@pytest.mark.unit
def test_idearium_append(tokenizer):
    """Test appending notions."""
    idearium = Idearium(tokenizer=tokenizer, max_tokens=20)

    # Append single notion
    notion = Notion(content="Hello", role=ChatRole.HUMAN)
    idearium.append(notion)
    assert len(idearium) == 1
    assert idearium[0] == notion

    # Append and combine same role notions
    notion2 = Notion(content=" World", role=ChatRole.HUMAN)
    idearium.append(notion2)
    assert len(idearium) == 1
    assert idearium[0].content == "Hello World"

    # Append different role (should not combine)
    notion3 = Notion(content="Hi", role=ChatRole.AI)
    idearium.append(notion3)
    assert len(idearium) == 2
    assert idearium[1] == notion3


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.idearium
@pytest.mark.unit
def test_idearium_token_limit(tokenizer):
    """Test token limit handling."""
    idearium = Idearium(tokenizer=tokenizer, max_tokens=5)

    # Add notions until limit
    idearium.append(Notion(content="12345", role=ChatRole.HUMAN))
    assert len(idearium) == 1

    # Adding more should remove the old notion and keep the new one
    idearium.append(Notion(content="123", role=ChatRole.AI))
    assert len(idearium) == 1
    assert idearium[0].role == str(ChatRole.AI.value)
    assert idearium[0].content == "123"  # New content is kept as is
    assert idearium.total_tokens == 3  # Verify token count


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.idearium
@pytest.mark.unit
def test_idearium_persistent_notions(tokenizer):
    """Test persistent notions handling."""
    idearium = Idearium(tokenizer=tokenizer, max_tokens=10)

    # Add persistent notion
    persistent = Notion(content="System", role=ChatRole.SYSTEM, persistent=True)
    idearium.append(persistent)

    # Add non-persistent notion
    regular = Notion(content="Hello", role=ChatRole.HUMAN)
    idearium.append(regular)

    # Add another that would exceed limit
    idearium.append(Notion(content="World", role=ChatRole.AI))

    # Verify persistent notion remains
    assert len(idearium) == 2
    assert idearium[0] == persistent


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.idearium
@pytest.mark.unit
def test_idearium_operations(tokenizer):
    """Test various Idearium operations."""
    idearium = Idearium(tokenizer=tokenizer, max_tokens=20)
    n1 = Notion(content="First", role=ChatRole.HUMAN)
    n2 = Notion(content="Second", role=ChatRole.AI)
    n3 = Notion(content="Third", role=ChatRole.HUMAN)

    # Test insert
    idearium.insert(0, n1)
    assert idearium[0] == n1

    # Test extend
    idearium.extend([n2, n3])
    assert len(idearium) == 3

    # Test remove
    idearium.remove(n2)
    assert len(idearium) == 2
    assert n2 not in idearium

    # Test pop
    popped = idearium.pop(0)
    assert popped == n1
    assert len(idearium) == 1

    # Test replace
    n4 = Notion(content="Fourth", role=ChatRole.AI)
    idearium.replace(0, n4)
    assert idearium[0] == n4


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.idearium
@pytest.mark.unit
def test_idearium_iteration(tokenizer):
    """Test iteration and container operations."""
    notions = [
        Notion(content="One", role=ChatRole.HUMAN),
        Notion(content="Two", role=ChatRole.AI),
        Notion(content="Three", role=ChatRole.HUMAN),
    ]
    idearium = Idearium(tokenizer=tokenizer, max_tokens=20, notions=notions)

    # Test iteration
    for i, notion in enumerate(idearium):
        assert notion == notions[i]

    # Test containment
    assert notions[0] in idearium
    assert Notion(content="Four", role=ChatRole.AI) not in idearium

    # Test indexing
    assert idearium[1] == notions[1]
    assert idearium.index(notions[2]) == 2


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.idearium
@pytest.mark.unit
def test_idearium_validation(tokenizer):
    """Test validation rules."""
    # Empty content should fail
    with pytest.raises(ValueError):
        Idearium(
            tokenizer=tokenizer,
            max_tokens=10,
            notions=[Notion(content="", role=ChatRole.HUMAN)],
        )

    # Notion exceeding max tokens should fail
    with pytest.raises(ValueError):
        Idearium(
            tokenizer=tokenizer,
            max_tokens=5,
            notions=[Notion(content="Too long!", role=ChatRole.HUMAN)],
        )

    # All persistent notions exceeding max tokens should fail
    with pytest.raises(ValueError):
        idearium = Idearium(tokenizer=tokenizer, max_tokens=5)
        idearium.append(Notion(content="123456", role=ChatRole.SYSTEM, persistent=True))
