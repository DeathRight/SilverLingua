from ....util import ImmutableMixin
from .role import RoleMember


class ReactRole(ImmutableMixin):
    THOUGHT = RoleMember("THOUGHT", "Thought")
    OBSERVATION = RoleMember("OBSERVATION", "Obs")
    ACTION = RoleMember("ACTION", "Act")
    QUESTION = RoleMember("QUESTION", "Q")
    ANSWER = RoleMember("ANSWER", "A")

    def __init__(
        self,
        THOUGHT="Thought",
        OBSERVATION="Obs",
        ACTION="Act",
        QUESTION="Q",
        ANSWER="A",
    ):
        object.__setattr__(self, "THOUGHT", RoleMember("THOUGHT", THOUGHT))
        object.__setattr__(self, "OBSERVATION", RoleMember("OBSERVATION", OBSERVATION))
        object.__setattr__(self, "ACTION", RoleMember("ACTION", ACTION))
        object.__setattr__(self, "QUESTION", RoleMember("QUESTION", QUESTION))
        object.__setattr__(self, "ANSWER", RoleMember("ANSWER", ANSWER))
