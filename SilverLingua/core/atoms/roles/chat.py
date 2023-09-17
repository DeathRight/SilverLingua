from ....util import ImmutableMixin
from .role import RoleMember


class ChatRole(ImmutableMixin):
    SYSTEM = RoleMember("SYSTEM", "SYSTEM")
    HUMAN = RoleMember("HUMAN", "HUMAN")
    AI = RoleMember("AI", "AI")
    TOOL_CALL = RoleMember("TOOL_CALL", "TOOL_CALL")
    TOOL_RESPONSE = RoleMember("TOOL_RESPONSE", "TOOL_RESPONSE")

    def __init__(
        self,
        SYSTEM="SYSTEM",
        HUMAN="HUMAN",
        AI="AI",
        TOOL_CALL="TOOL_CALL",
        TOOL_RESPONSE="TOOL_RESPONSE",
    ):
        object.__setattr__(self, "SYSTEM", RoleMember("SYSTEM", SYSTEM))
        object.__setattr__(self, "HUMAN", RoleMember("HUMAN", HUMAN))
        object.__setattr__(self, "AI", RoleMember("AI", AI))
        object.__setattr__(self, "TOOL_CALL", RoleMember("TOOL_CALL", TOOL_CALL))
        object.__setattr__(
            self, "TOOL_RESPONSE", RoleMember("TOOL_RESPONSE", TOOL_RESPONSE)
        )


OpenAIChatRole = ChatRole(
    SYSTEM="system",
    HUMAN="user",
    AI="assistant",
    TOOL_CALL="function_call",
    TOOL_RESPONSE="function",
)
