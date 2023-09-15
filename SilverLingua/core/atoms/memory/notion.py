from typing import Any

from memory import Memory

from ....constants import config
from ..roles import ChatRole, ReactRole


class Notion(Memory):
    """
    A memory that stores the role associated with it's content.

    The role is usually 'SYSTEM', 'HUMAN', 'AI', etc.
    (See `config`)

    However, it can also be a ReAct role, such as
    'Thought', 'Observation', 'Action', etc.
    """

    role: str

    def __init__(self, content: str, role: Any) -> None:
        super().__init__(content)

        if isinstance(role, ChatRole) or (
            hasattr(role, "name") and hasattr(role, "value")
        ):
            self.role = role.value
        elif isinstance(role, str):
            self.role = role
        else:
            raise TypeError(f"Expected an enum member or a string, got {type(role)}")

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def __repr__(self) -> str:
        return f"Notion({self.content}, {self.role})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Notion):
            return NotImplemented
        return self.content == other.content and self.role == other.role

    @property
    def chat_role(self) -> ChatRole:
        """
        Gets the chat based role Enum (e.g. Role.SYSTEM, Role.HUMAN, etc.)

        (See `config`)
        """
        # Check if self.role is a member of Role
        r = config.get_chat_role(self.role)
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
        # Check if self.role is a member of Role
        r = config.get_react_role(self.role)
        if r is None:
            # If not, then the role is AI.
            # Why? Because it must be an internal role.
            return ReactRole.THOUGHT
        return r
