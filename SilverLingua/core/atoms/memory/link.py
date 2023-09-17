from typing import List, Optional, Union

from .memory import Memory
from .notion import Notion


class Link(Memory):
    """
    A memory that can have a parent and children Links,
    forming a hierarchical structure of interconnected memories.

    The content can be either a Notion or a Memory.
    (You can still use the content as a string)
    """

    parent: Optional["Link"]
    children: List["Link"]
    content: Union[Notion, Memory]

    def __init__(
        self,
        content: Union[Notion, Memory],
        parent: Optional["Link"] = None,
        children: List["Link"] = None,
    ) -> None:
        if children is None:
            children = []
        self.content = content
        self.parent = parent
        self.children = children if children is not None else []

    def add_child(self, child: "Link") -> None:
        self.children.append(child)
        child.parent = self

    def remove_child(self, child: "Link") -> None:
        self.children.remove(child)
        child.parent = None

    def __str__(self) -> str:
        return f"Link(content={self.content}, parent={self.parent}, children={self.children})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Link):
            return NotImplemented
        return (
            self.content == other.content
            and self.parent == other.parent
            and self.children == other.children
        )
