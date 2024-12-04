from typing import Union

from pydantic import ConfigDict, field_validator

from ..atoms.memory import Memory
from ..atoms.role import ChatRole, ReactRole


class Notion(Memory):
    """
    A memory that stores the role associated with its content.
    The role is usually a [`ChatRole`][silverlingua.core.atoms.role.chat.ChatRole] or a [`ReactRole`][silverlingua.core.atoms.role.react.ReactRole].
    (See [`atoms/roles`][silverlingua.core.atoms.role])

    Attributes:
        role: The role of the notion.
        content: The content of the memory.
        persistent: Whether the notion should be stored in long-term memory.
    """

    model_config = ConfigDict(from_attributes=True)
    role: str
    persistent: bool = False

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v: Union[ChatRole, ReactRole, str]):
        if isinstance(v, (ChatRole, ReactRole)):
            return v.value.value
        elif isinstance(v, str):
            return v
        raise ValueError(f"Expected a ChatRole, ReactRole, or a string, got {type(v)}")

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def __init__(
        self,
        content: str,
        role: Union[ChatRole, ReactRole, str],
        persistent: bool = False,
    ):
        super().__init__(content=content, role=role, persistent=persistent)

    @property
    def chat_role(self) -> ChatRole:
        """
        Gets the chat based role Enum (e.g. Role.SYSTEM, Role.HUMAN, etc.)

        (See `config`)
        """
        from ...config import Config

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
        from ...config import Config

        # Check if self.role is a member of Role
        r = Config.get_react_role(self.role)
        if r is None:
            # If not, then the role is AI.
            # Why? Because it must be an internal role.
            return ReactRole.THOUGHT
        return r
