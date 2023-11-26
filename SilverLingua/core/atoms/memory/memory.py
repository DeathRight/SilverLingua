from pydantic import BaseModel


class Memory(BaseModel):
    """
    A Memory is the smallest unit of storage information, and
    is the base class for all other storage information.
    """

    content: str

    class Config:
        # Configuring Pydantic's BaseModel behavior
        anystr_strip_whitespace = True
        min_anystr_length = 1

    def __str__(self) -> str:
        return self.content
