from ....util import ImmutableAttributeError


class RoleMember:
    """
    Used in ChatRole and ReactRole.

    Warning:
        **Do not** instantiate this class directly. See [ChatRole][SilverLingua.core.atoms.role.chat.ChatRole] and [ReactRole][SilverLingua.core.atoms.role.react.ReactRole] for more information.
    """

    def __init__(self, name, value, parent=None):
        self.__dict__["_name"] = name
        self.__dict__["_value"] = value
        self.__dict__["_parent"] = parent

    @property
    def name(self) -> str:
        return self._name  # type: ignore

    @property
    def value(self) -> str:
        return self._value  # type: ignore

    def __setattr__(self, name, value):
        if name == "_parent" and self.__dict__.get("_parent") is None:
            self.__dict__[name] = value
            return
        raise ImmutableAttributeError("RoleMember attributes are immutable.")

    def __eq__(self, other):
        return (
            isinstance(other, RoleMember)
            and other._parent == self._parent
            and other.name == self.name
        )  # type: ignore

    def __str__(self):
        return self.value
