from typing import List

import pytest
from pydantic import ValidationError

from silverlingua.core.atoms import Tokenizer


def test_tokenizer_initialization():
    """Test that Tokenizer can be initialized with encode/decode functions."""

    def mock_encode(text: str) -> List[int]:
        return [ord(c) for c in text]

    def mock_decode(tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)

    tokenizer = Tokenizer(encode=mock_encode, decode=mock_decode)
    assert callable(tokenizer.encode)
    assert callable(tokenizer.decode)


def test_tokenizer_encode():
    """Test encoding functionality."""

    def mock_encode(text: str) -> List[int]:
        return [ord(c) for c in text]

    def mock_decode(tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)

    tokenizer = Tokenizer(encode=mock_encode, decode=mock_decode)

    # Test empty string
    assert tokenizer.encode("") == []

    # Test simple string
    assert tokenizer.encode("hello") == [104, 101, 108, 108, 111]

    # Test string with spaces
    assert tokenizer.encode("hello world") == [
        104,
        101,
        108,
        108,
        111,
        32,
        119,
        111,
        114,
        108,
        100,
    ]

    # Test special characters
    assert tokenizer.encode("!@#$%") == [33, 64, 35, 36, 37]


def test_tokenizer_decode():
    """Test decoding functionality."""

    def mock_encode(text: str) -> List[int]:
        return [ord(c) for c in text]

    def mock_decode(tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)

    tokenizer = Tokenizer(encode=mock_encode, decode=mock_decode)

    # Test empty list
    assert tokenizer.decode([]) == ""

    # Test simple tokens
    assert tokenizer.decode([104, 101, 108, 108, 111]) == "hello"

    # Test tokens with space
    assert (
        tokenizer.decode([104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100])
        == "hello world"
    )

    # Test special characters
    assert tokenizer.decode([33, 64, 35, 36, 37]) == "!@#$%"


def test_tokenizer_roundtrip():
    """Test that encoding and then decoding returns the original input."""

    def mock_encode(text: str) -> List[int]:
        return [ord(c) for c in text]

    def mock_decode(tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)

    tokenizer = Tokenizer(encode=mock_encode, decode=mock_decode)

    test_strings = [
        "",  # Empty string
        "hello",  # Simple string
        "hello world",  # String with space
        "Hello, World!",  # String with punctuation
        "1234567890",  # Numbers
        "!@#$%^&*()",  # Special characters
        "Mixed 123 Content !@#",  # Mixed content
        "Unicode ♥ ☺ ♦",  # Unicode characters
    ]

    for test_str in test_strings:
        tokens = tokenizer.encode(test_str)
        decoded = tokenizer.decode(tokens)
        assert decoded == test_str


def test_tokenizer_validation():
    """Test input validation and error cases."""

    def mock_encode(text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        return [ord(c) for c in text]

    def mock_decode(tokens: List[int]) -> str:
        if not isinstance(tokens, list):
            raise TypeError("Input must be a list")
        if not all(isinstance(t, int) for t in tokens):
            raise TypeError("All tokens must be integers")
        return "".join(chr(t) for t in tokens)

    # Test initialization with invalid encode function
    with pytest.raises(ValidationError):
        Tokenizer(encode="not_a_function", decode=mock_decode)

    # Test initialization with invalid decode function
    with pytest.raises(ValidationError):
        Tokenizer(encode=mock_encode, decode="not_a_function")

    tokenizer = Tokenizer(encode=mock_encode, decode=mock_decode)

    # Test encode with non-string input
    with pytest.raises(TypeError, match="Input must be a string"):
        tokenizer.encode(123)  # type: ignore

    # Test decode with non-list input
    with pytest.raises(TypeError, match="Input must be a list"):
        tokenizer.decode("not_a_list")  # type: ignore

    # Test decode with list of non-ints
    with pytest.raises(TypeError, match="All tokens must be integers"):
        tokenizer.decode(["not", "integers"])  # type: ignore

    # Test decode with invalid token values
    with pytest.raises(ValueError):
        tokenizer.decode([-1])  # Invalid character code
