import os

import pytest
from dotenv import load_dotenv

from silverlingua.core.molecules.notion import Notion
from silverlingua.core.organisms import Idearium

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    try:
        from silverlingua_anthropic import Anthropic

        return Anthropic(api_key=api_key)
    except ImportError:
        pytest.skip("Anthropic package not installed")


@pytest.fixture
def tokenizer(anthropic_client):
    try:
        from silverlingua_anthropic.templates.model.tokenizer import AnthropicTokenizer

        return AnthropicTokenizer(
            client=anthropic_client, model="claude-3-opus-20240229"
        )
    except ImportError:
        pytest.skip("Anthropic package not installed")


@pytest.mark.anthropic
def test_tokenizer_encode(tokenizer):
    """Test basic encoding functionality."""
    text = "Hello, world!"
    tokens = tokenizer.encode(text)

    # Verify we get a list of integers
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    # Verify cache works
    cached_tokens = tokenizer.encode(text)
    assert tokens == cached_tokens


@pytest.mark.anthropic
def test_tokenizer_encode_long(tokenizer):
    """Test encoding a long text."""
    long_text = "This is a very long text that should exceed our token limit. " * 50
    tokens = tokenizer.encode(long_text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0


@pytest.mark.anthropic
def test_tokenizer_decode(tokenizer):
    """Test basic decoding functionality."""
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    # For exact matches, we should get the original text
    assert decoded == text


@pytest.mark.anthropic
def test_tokenizer_partial_decode(tokenizer):
    """Test decoding of partial token sequences."""
    text = "This is a test of partial decoding capabilities."
    tokens = tokenizer.encode(text)

    # Take a subset of tokens
    partial_tokens = tokens[: len(tokens) // 2]
    decoded = tokenizer.decode(partial_tokens)

    # Verify we get a non-empty string back
    assert decoded
    assert isinstance(decoded, str)


@pytest.mark.anthropic
def test_tokenizer_cache(tokenizer):
    """Test the tokenizer's caching behavior."""
    text1 = "First test string"
    text2 = "Second test string"

    # Encode both strings
    tokens1 = tokenizer.encode(text1)
    tokens2 = tokenizer.encode(text2)

    # Verify different texts get different tokens
    assert tokens1 != tokens2

    # Verify cache hits return same tokens
    assert tokenizer.encode(text1) == tokens1
    assert tokenizer.encode(text2) == tokens2


@pytest.mark.anthropic
def test_tokenizer_with_idearium(tokenizer):
    """Test the tokenizer works with Idearium."""
    idearium = Idearium(
        tokenizer=tokenizer,
        max_tokens=100,
        notions=[
            Notion(content="Hello", role="user"),
        ],
    )

    # Test appending a notion with same role (should combine)
    idearium.append(Notion(content=" How are you?", role="user"))

    # Test appending a notion with different role
    idearium.append(Notion(content="Hi there!", role="assistant"))

    # Verify token counting works
    assert idearium.total_tokens > 0

    # Test trimming works
    long_text = "This is a very long text that should exceed our token limit. " * 50
    idearium.append(Notion(content=long_text, role="assistant"))

    # Verify the idearium was trimmed
    assert idearium.total_tokens <= 100
