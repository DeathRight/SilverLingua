from enum import Enum
from typing import Type

from .member import RoleMember


class ChatRole(Enum):
    SYSTEM = RoleMember("SYSTEM", "SYSTEM")
    HUMAN = RoleMember("HUMAN", "HUMAN")
    AI = RoleMember("AI", "AI")
    TOOL_CALL = RoleMember("TOOL_CALL", "TOOL_CALL")
    TOOL_RESPONSE = RoleMember("TOOL_RESPONSE", "TOOL_RESPONSE")

    def __eq__(self, other):
        if not isinstance(other, Enum):
            return NotImplemented
        return self.value == other.value


# Set the parent of each member to ChatRole
for member in ChatRole:
    member.value._parent = ChatRole


def create_chat_role(
    name: str, SYSTEM: str, HUMAN: str, AI: str, TOOL_CALL: str, TOOL_RESPONSE: str
) -> Type[ChatRole]:
    """
    Create a new ChatRole enum with only the values of the RoleMembers changed.

    This will ensure that the parent of each member is ChatRole, which means
    that the members will be equal to the members of ChatRole.

    Example:
        ```python
        OpenAIChatRole = create_chat_role(
            "OpenAIChatRole",
            SYSTEM="system",
            HUMAN="user",
            AI="assistant",
            TOOL_CALL="function_call",
            TOOL_RESPONSE="function",
        )

        assert OpenAIChatRole.SYSTEM == ChatRole.SYSTEM # True
        ```
    """
    return Enum(
        name,
        {
            "SYSTEM": RoleMember("SYSTEM", SYSTEM, ChatRole),
            "HUMAN": RoleMember("HUMAN", HUMAN, ChatRole),
            "AI": RoleMember("AI", AI, ChatRole),
            "TOOL_CALL": RoleMember("TOOL_CALL", TOOL_CALL, ChatRole),
            "TOOL_RESPONSE": RoleMember("TOOL_RESPONSE", TOOL_RESPONSE, ChatRole),
        },
    )
