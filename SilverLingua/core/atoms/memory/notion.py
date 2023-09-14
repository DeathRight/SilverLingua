from enum import Enum
from typing import Any, Union

from memory import Memory

from ....config import ChatRole
from ....constants import config


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

    if isinstance(role, ChatRole) or (hasattr(role, 'name')
                                      and hasattr(role, 'value')):
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
      Get's the chat based role Enum (e.g. Role.SYSTEM, Role.HUMAN, etc.)
      
      (See `config`)
      """
    # Check if self.role is a member of Role
    r = config.get_chat_role(self.role)
    if r is None:
      # If not, then the role is SYSTEM.
      # Why? Because it must be an internal role.
      return ChatRole.SYSTEM
    return r
