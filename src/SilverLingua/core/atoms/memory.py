from pydantic import BaseModel


class Memory(BaseModel):
    """
    A Memory is the smallest unit of storage information, and
    is the base class for all other storage information like [`Notion`][silverlingua.core.molecules.notion.Notion].
    """

    content: str

    def __str__(self) -> str:
        return self.content
