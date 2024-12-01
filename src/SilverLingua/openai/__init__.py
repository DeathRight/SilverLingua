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
