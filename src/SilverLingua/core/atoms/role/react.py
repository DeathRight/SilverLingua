from enum import Enum
from typing import Type

from .member import RoleMember


class ReactRole(Enum):
    THOUGHT = RoleMember("THOUGHT", "THOUGHT")
    OBSERVATION = RoleMember("OBSERVATION", "OBSERVATION")
    ACTION = RoleMember("ACTION", "ACTION")
    QUESTION = RoleMember("QUESTION", "QUESTION")
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
    THOUGHT: str,
    OBSERVATION: str,
    ACTION: str,
    QUESTION: str,
    ANSWER: str,
) -> Type[ReactRole]:
    return Enum(
        name,
        {
            "THOUGHT": RoleMember("THOUGHT", THOUGHT, ReactRole),
            "OBSERVATION": RoleMember("OBSERVATION", OBSERVATION, ReactRole),
            "ACTION": RoleMember("ACTION", ACTION, ReactRole),
            "QUESTION": RoleMember("QUESTION", QUESTION, ReactRole),
            "ANSWER": RoleMember("ANSWER", ANSWER, ReactRole),
        },
    )
