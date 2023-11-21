import logging

from SilverLingua import Module

from .atoms import ChatRole, ReactRole

logger = logging.getLogger(__name__)

core_module = Module(
    name="core",
    description="The core module of SilverLingua.",
    version="1.0.0",
    tools=[],
    chat_roles=[ChatRole],
    react_roles=[ReactRole],
)
