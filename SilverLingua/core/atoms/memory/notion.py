from typing import Union

from pydantic import validator

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

    @validator("role", pre=True, always=True)
    def validate_role(cls, v: Union[ChatRole, ReactRole, str]):
        if isinstance(v, (ChatRole, ReactRole)):
            return v.value
        elif isinstance(v, str):
            return v
        raise TypeError(f"Expected a ChatRole, ReactRole, or a string, got {type(v)}")

    class Config:
        orm_mode = True

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

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
