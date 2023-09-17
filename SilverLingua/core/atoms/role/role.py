from ....util import ImmutableAttributeError


class RoleMember:
    def __init__(self, name, value, parent=None):
        self.__dict__["_name"] = name
        self.__dict__["_value"] = value
        self.__dict__["_parent"] = parent or self.__class__

    @property
    def name(self):
        return self._name  # type: ignore

    @property
    def value(self):
        return self._value  # type: ignore

    def __setattr__(self, name, value):
        raise ImmutableAttributeError("RoleMember attributes are immutable.")

    def __eq__(self, other):
        return (
            isinstance(other, RoleMember)
            and other._parent is self._parent
            and other.name == self.name
        )  # type: ignore

    def __str__(self):
        return self.value
