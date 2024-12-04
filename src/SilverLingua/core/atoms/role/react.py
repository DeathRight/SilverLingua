from enum import Enum
from typing import Type

from .member import RoleMember


class ReactRole(Enum):
    """
    Standardized roles for ReAct framework to help with chain-of-thought prompting.

    See: [https://www.promptingguide.ai/techniques/react](https://www.promptingguide.ai/techniques/react)

    Warning:
        **Do not** instantiate this enum directly. Use [`create_react_role()`][silverlingua.core.atoms.role.react.create_react_role] instead.
    """

    QUESTION = RoleMember("QUESTION", "QUESTION")
    THOUGHT = RoleMember("THOUGHT", "THOUGHT")
    ACTION = RoleMember("ACTION", "ACTION")
    OBSERVATION = RoleMember("OBSERVATION", "OBSERVATION")
    ANSWER = RoleMember("ANSWER", "ANSWER")

    def __eq__(self, other):
        if not isinstance(other, Enum):
            return NotImplemented
        return self.value == other.value


# Set the parent of each member to ReactRole
for member in ReactRole:
    member.value._parent = ReactRole


def create_react_role(
    name: str,
    QUESTION: str,
    THOUGHT: str,
    ACTION: str,
    OBSERVATION: str,
    ANSWER: str,
) -> Type[ReactRole]:
    return Enum(
        name,
        {
            "QUESTION": RoleMember("QUESTION", QUESTION, ReactRole),
            "THOUGHT": RoleMember("THOUGHT", THOUGHT, ReactRole),
            "ACTION": RoleMember("ACTION", ACTION, ReactRole),
            "OBSERVATION": RoleMember("OBSERVATION", OBSERVATION, ReactRole),
            "ANSWER": RoleMember("ANSWER", ANSWER, ReactRole),
        },
    )
