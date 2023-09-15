from ....util import MatchingNameEnum


class ReactRole(MatchingNameEnum):
    THOUGHT = "Thought"
    OBSERVATION = "Obs"
    ACTION = "Act"
    QUESTION = "Q"
    ANSWER = "A"
