"""
SilverLingua - An AI agent framework
"""

import logging

# Finally, import provider implementations
from .anthropic import AnthropicChatAgent, AnthropicChatRole, AnthropicModel

# First, import the config module
from .config import Config, Module

# Then import core components
from .core.atoms import (
    ChatRole,
    Memory,
    ReactRole,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolCallResponse,
    ToolCalls,
    tool,
)
from .core.molecules import Link, Notion
from .core.organisms import Idearium
from .core.templates import Agent, Model
from .openai import OpenAIChatAgent, OpenAIChatRole, OpenAIModel

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__ = [
    # Core components
    "ChatRole",
    "Memory",
    "ReactRole",
    "Tool",
    "ToolCall",
    "ToolCallFunction",
    "ToolCallResponse",
    "ToolCalls",
    "tool",
    "Link",
    "Notion",
    "Idearium",
    "Agent",
    "Model",
    # OpenAI implementation
    "OpenAIChatAgent",
    "OpenAIChatRole",
    "OpenAIModel",
    # Anthropic implementation
    "AnthropicChatAgent",
    "AnthropicChatRole",
    "AnthropicModel",
    # Configuration
    "Config",
    "Module",
]
