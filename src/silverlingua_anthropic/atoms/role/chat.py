from silverlingua.core.atoms import create_chat_role

AnthropicChatRole = create_chat_role(
    "AnthropicChatRole",
    SYSTEM="system",
    HUMAN="user",
    AI="assistant",
    TOOL_CALL="assistant",  # Anthropic uses assistant role for tool calls
    TOOL_RESPONSE="tool",  # Tool responses are marked as tool role
)
