from typing import List, Optional, Union

from .memory import Memory
from .notion import Notion


class Link(Memory):
    """
    A memory that can have a parent and children Links,
    forming a hierarchical structure of interconnected memories.

    The content can be either a Notion or a Memory.
    (You can still use the content as a string via str(link.content).
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

    @property
    def path(self) -> List["Link"]:
        """
        Returns the path from the root to this Link.
        """
        path = [self]
        while path[-1].parent is not None:
            path.append(path[-1].parent)
        return path

    @property
    def root(self) -> "Link":
        """
        Returns the root Link of this Link.
        """
        return self.path[-1]

    @property
    def depth(self) -> int:
        """
        Returns 1 based depth of this Link.
        """
        return len(self.path)

    @property
    def is_root(self) -> bool:
        """
        Returns whether this Link is a root Link.
        """
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """
        Returns whether this Link is a leaf Link.
        """
        return len(self.children) == 0

    @property
    def is_branch(self) -> bool:
        """
        Returns whether this Link is a branch Link.
        """
        return not self.is_leaf

    @property
    def path_string(self) -> str:
        """
        Returns the path from the root to this Link as a string.

        Example:
        "root>child>grandchild"
        """
        path_str = f"{self.content}"
        if not self.is_root:
            path_str = f"{self.parent.path_string}>{path_str}"
        return path_str

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
