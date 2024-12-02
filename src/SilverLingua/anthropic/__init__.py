"""
The Anthropic module provides implementations of SilverLingua's core components
using the Anthropic API.

This module includes:
- AnthropicChatAgent: An agent that uses Anthropic's chat completion API
- AnthropicModel: A model that uses Anthropic's API
- AnthropicChatRole: Role definitions for Anthropic's chat format
"""

import logging

from ..config import Module
from .atoms import AnthropicChatRole
from .templates.agent import AnthropicChatAgent
from .templates.model import AnthropicModel

__all__ = ["AnthropicChatAgent", "AnthropicModel", "AnthropicChatRole"]

logger = logging.getLogger(__name__)

anthropic_module = Module(
    name="Anthropic",
    description="Adds Anthropic models and agents.",
    version="1.0.0",
    tools=[],
    chat_roles=[AnthropicChatRole],
    react_roles=[],
)
