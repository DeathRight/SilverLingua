import logging

from SilverLingua import Module

from .agent import OpenAIChatAgent  # noqa
from .atoms import OpenAIChatRole

logger = logging.getLogger(__name__)

openai_module = Module(
    name="OpenAI",
    description="Adds OpenAI models and agents.",
    version="1.0.0",
    tools=[],
    chat_roles=[OpenAIChatRole],
    react_roles=[],
)
