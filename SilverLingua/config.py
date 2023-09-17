from typing import List, Optional

from .core.atoms.roles import ChatRole, OpenAIChatRole, ReactRole
from .core.atoms.tool import Tool


class Config:
    chat_roles: List[ChatRole]
    react_roles: List[ReactRole]
    tools: List[Tool]

    def __init__(self):
        self.chat_roles = [ChatRole, OpenAIChatRole]
        self.react_roles = [ReactRole]

    def get_chat_role(self, role: str) -> Optional[ChatRole]:
        """
        Attempts to get the standardized ChatRole enum from 'role'.
        If not, returns None.

        This is usually used internally for maintaining
        consistency in Notions across different LLM backends.
        """
        for enum_class in self.chat_roles:
            for enum_member in enum_class:
                if enum_member.value == role:
                    return ChatRole[enum_member.name]
        return None

    def get_react_role(self, role: str) -> Optional[ReactRole]:
        """
        Attempts to get the standardized ReactRole enum from 'role'.
        If not, returns None.

        This is usually used internally for maintaining
        consistency in Notions across different LLM backends.
        """
        for enum_class in self.react_roles:
            for enum_member in enum_class:
                if enum_member.value == role:
                    return ReactRole[enum_member.name]
        return None
