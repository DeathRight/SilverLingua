"""
The OpenAI module provides implementations of SilverLingua's core components
using the OpenAI API.

This module includes:
- OpenAIChatAgent: An agent that uses OpenAI's chat completion API
- OpenAIModel: A model that uses OpenAI's API
- OpenAIChatRole: Role definitions for OpenAI's chat format
"""

import logging

from SilverLingua import Module

from .atoms import OpenAIChatRole
from .templates.agent import OpenAIChatAgent
from .templates.model import OpenAIModel

__all__ = ["OpenAIChatAgent", "OpenAIChatRole", "OpenAIModel"]
logger = logging.getLogger(__name__)

openai_module = Module(
    name="OpenAI",
    description="Adds OpenAI models and agents.",
    version="1.0.0",
    tools=[],
    chat_roles=[OpenAIChatRole],
    react_roles=[],
)
