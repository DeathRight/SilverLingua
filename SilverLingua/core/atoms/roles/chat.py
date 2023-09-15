from ....util import MatchingNameEnum


# A base enum for chat roles. Any derived enum must have the same names.
class ChatRole(MatchingNameEnum):
  """
    A base Enum representing various chat roles. When subclassing, ensure that the 
    new Enum has exactly the same names as this base Enum for interoperability.
    """
  SYSTEM = "SYSTEM"
  HUMAN = "HUMAN"
  AI = "AI"
  TOOL_CALL = "TOOL_CALL"
  TOOL_RESPONSE = "TOOL_RESPONSE"


# This will succeed because the names match the base enum
class OpenAIChatRole(ChatRole):  # type: ignore
  """
    A derived Enum from ChatRole. Maps the standard chat roles to OpenAI-specific roles.
    The names are forced to match those in ChatRole for interoperability.
    """
  SYSTEM = "system"
  HUMAN = "user"
  AI = "assistant"
  TOOL_CALL = "function_call"
  TOOL_RESPONSE = "function"
