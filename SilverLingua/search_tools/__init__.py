import logging

from SilverLingua import Module

from .google_search import google_search

logger = logging.getLogger(__name__)

search_tools_module = Module(
    name="Search Tools",
    description="Adds various tools for searching the web.",
    version="1.0.0",
    tools=[google_search],
)
