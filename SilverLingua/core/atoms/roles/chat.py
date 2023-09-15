from ....util import MatchingNameEnum


# A base enum for chat roles. Any derived enum must have the same names.
class ChatRole(MatchingNameEnum):
    """
    A base Enum representing various chat roles. When subclassing, ensure that the
    new Enum has exactly the same names as this base Enum for interoperability.

    Raises:
        TypeError: If a subclass does not have the same names as this base Enum.

    Example:
        ```python
            class MyChatRole(ChatRole):
                SYSTEM = "system"
                HUMAN = "user"
                AI = "ai"
                TOOL_CALL = "TOOL_CALL"
                TOOL_RESPONSE = "TOOL_RESPONSE"
        ```
        This is ok! The names match.

        ```python
            class MyChatRole(ChatRole):
                SYSTEM = "system"
                HUMAN = "user"
                AI = "ai"
                TOOL_CALL = "tool_call"
                TOOL = "tool_response"
        ```
        This is not ok! The names do not match. Will raise a TypeError.
    """

    SYSTEM = "SYSTEM"
    HUMAN = "HUMAN"
    AI = "AI"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESPONSE = "TOOL_RESPONSE"


class OpenAIChatRole(ChatRole):
    """
    A derived Enum from ChatRole. Maps the standard chat roles to OpenAI-specific roles.
    The names are forced to match those in ChatRole for interoperability.
    """

    SYSTEM = "system"
    HUMAN = "user"
    AI = "assistant"
    TOOL_CALL = "function_call"
    TOOL_RESPONSE = "function"
