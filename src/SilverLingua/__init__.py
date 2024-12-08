"""
SilverLingua - An AI agent framework
"""

import logging

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
    # Configuration
    "Config",
    "Module",
]
