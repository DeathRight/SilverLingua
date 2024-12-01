"""
SilverLingua - An AI agent framework

SilverLingua is a framework for building AI agents that can interact with tools,
maintain memory, and engage in complex conversations.

Core Components:
- Atoms: Basic building blocks (Memory, Tokenizer, Tool, etc.)
- Molecules: Simple compositions (Notion, Link)
- Organisms: Complex compositions (Idearium)
- Templates: Base classes for models and agents

Implementations:
- OpenAI: OpenAI-specific implementations
- Anthropic: Anthropic-specific implementations
"""

import logging

# Anthropic implementation
from .anthropic import AnthropicChatAgent, AnthropicChatRole, AnthropicModel
from .config import Config, Module  # noqa: F401

# Core components
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

# OpenAI implementation
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
