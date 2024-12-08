from silverlingua.core.atoms import create_chat_role

OpenAIChatRole = create_chat_role(
    "OpenAIChatRole",
    SYSTEM="system",
    HUMAN="user",
    AI="assistant",
    TOOL_CALL="assistant",
    TOOL_RESPONSE="tool",
)
