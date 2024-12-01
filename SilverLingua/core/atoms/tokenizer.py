from typing import Callable, List

from pydantic import BaseModel


class Tokenizer(BaseModel):
    """
    A tokenizer that can encode and decode strings.
    """

    encode: Callable[[str], List[int]]
    decode: Callable[[List[int]], str]
