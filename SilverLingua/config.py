from enum import Enum
from typing import List, Optional, Type


class ChatRole(Enum):
  SYSTEM = "SYSTEM"
  HUMAN = "HUMAN"
  AI = "AI"
  TOOL_CALL = "TOOL_CALL"
  TOOL_RESPONSE = "TOOL_RESPONSE"


class OpenAIChatRole(Enum):
  SYSTEM = "system"
  HUMAN = "user"
  AI = "assistant"
  TOOL_CALL = "function_call"
  TOOL_RESPONSE = "function"


class Config:
  chat_roles: List[Type[Enum]]

  def __init__(self):
    self.chat_roles = [ChatRole, OpenAIChatRole]

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
