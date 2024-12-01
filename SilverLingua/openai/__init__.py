import logging

from SilverLingua import Module

from .atoms import OpenAIChatRole
from .templates.agent import OpenAIChatAgent

__all__ = ["OpenAIChatAgent", "OpenAIChatRole"]
logger = logging.getLogger(__name__)

openai_module = Module(
    name="OpenAI",
    description="Adds OpenAI models and agents.",
    version="1.0.0",
    tools=[],
    chat_roles=[OpenAIChatRole],
    react_roles=[],
)
