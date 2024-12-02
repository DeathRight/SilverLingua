"""
Role-related functionality for SilverLingua.

This module provides enums and utilities for handling different types of roles in conversations:
- Chat roles (system, human, AI, etc.)
- React roles (thought, observation, action, etc.)
"""

from .chat import ChatRole, create_chat_role
from .member import RoleMember
from .react import ReactRole, create_react_role

__all__ = [
    "RoleMember",
    "ChatRole",
    "ReactRole",
    "create_chat_role",
    "create_react_role",
]
