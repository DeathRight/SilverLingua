from typing import Any

from ..role import ChatRole, ReactRole
from .memory import Memory


class Notion(Memory):
    """
    A memory that stores the role associated with its content.

    The role is usually a `ChatRole` or a `ReactRole`.
    (See `atoms/roles`)

    Attributes:
        role: The role of the notion.
        content: The content of the memory.
        persistent: Whether the notion should be stored in long-term memory.
    """

    role: str
    persistent: bool = False

    def __init__(self, content: str, role: Any, persistent: bool = False) -> None:
        super().__init__(content)

        if hasattr(role, "name") and hasattr(role, "value"):
            self.role = role.value
        elif isinstance(role, str):
            self.role = role
        else:
            raise TypeError(f"Expected an enum member or a string, got {type(role)}")

        self.persistent = persistent

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def __repr__(self) -> str:
        return f"Notion({self.content}, {self.role}, {self.persistent})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Notion):
            return NotImplemented
        return (
            self.content == other.content
            and self.role == other.role
            and self.persistent == other.persistent
        )

    @property
    def chat_role(self) -> ChatRole:
        """
        Gets the chat based role Enum (e.g. Role.SYSTEM, Role.HUMAN, etc.)

        (See `config`)
        """
        from ....config import Config

        # Check if self.role is a member of Role
        r = Config.get_chat_role(self.role)
        if r is None:
            # If not, then the role is AI.
            # Why? Because it must be an internal role.
            return ChatRole.AI
        return r

    @property
    def react_role(self) -> ReactRole:
        """
        Gets the react based role Enum (e.g. Role.THOUGHT, Role.OBSERVATION, etc.)

        (See `config`)
        """
        from ....config import Config

        # Check if self.role is a member of Role
        r = Config.get_react_role(self.role)
        if r is None:
            # If not, then the role is AI.
            # Why? Because it must be an internal role.
            return ReactRole.THOUGHT
        return r
