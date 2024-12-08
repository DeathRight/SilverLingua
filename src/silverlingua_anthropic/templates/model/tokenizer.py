import logging
from typing import Dict, List

from anthropic import Anthropic
from pydantic import ConfigDict

from silverlingua.core.atoms import Tokenizer

logger = logging.getLogger(__name__)


class _TokenizerState:
    """Internal state for the AnthropicTokenizer."""

    def __init__(self, client: Anthropic, model: str):
        self.client = client
        self.model = model
        self.token_cache: Dict[str, List[int]] = {}


class AnthropicTokenizer(Tokenizer):
    """
    A tokenizer implementation for Anthropic models that uses the count_tokens API.

    Note: Due to limitations in Anthropic's API, this tokenizer:
    1. Uses the count_tokens API for accurate token counts
    2. Maintains an internal cache of text->token mappings
    3. Provides approximate decoding (best effort)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, client: Anthropic, model: str):
        """Initialize the tokenizer with an Anthropic client and model name."""
        # Initialize internal state
        state = _TokenizerState(client, model)

        # Define encode and decode functions
        def encode(text: str) -> List[int]:
            # Check cache first
            if text in state.token_cache:
                return state.token_cache[text]

            # Get token count from Anthropic API
            response = state.client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=state.model,
                messages=[{"role": "user", "content": text}],
            )

            # Generate synthetic token IDs
            # Note: These are not real Anthropic token IDs, but serve as placeholders
            # that maintain the correct count and can be decoded back to the original text
            token_count = response.input_tokens
            tokens = list(
                range(len(state.token_cache), len(state.token_cache) + token_count)
            )

            # Cache the mapping
            state.token_cache[text] = tokens

            return tokens

        def decode(tokens: List[int]) -> str:
            # Look up the original text from our cache
            for text, cached_tokens in state.token_cache.items():
                if tokens == cached_tokens:
                    return text

            # If we can't find an exact match, try to reconstruct from substrings
            # This is a best-effort approach when dealing with trimmed token sequences
            result = []
            for token in tokens:
                for text, cached_tokens in state.token_cache.items():
                    if token in cached_tokens:
                        idx = cached_tokens.index(token)
                        # Estimate the character length per token
                        chars_per_token = len(text) // len(cached_tokens)
                        # Extract the approximate substring
                        start = idx * chars_per_token
                        end = start + chars_per_token
                        result.append(text[start:end])
                        break
                else:
                    # If we can't find the token, append a placeholder
                    result.append(" ")

            return "".join(result)

        # Initialize base class with encode/decode functions
        super().__init__(encode=encode, decode=decode)
