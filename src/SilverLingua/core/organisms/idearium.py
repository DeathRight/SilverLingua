import logging
from typing import Iterator, List, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..atoms.tokenizer import Tokenizer
from ..molecules.notion import Notion

logger = logging.getLogger(__name__)


class Idearium(BaseModel):
    """
    A collection of `Notions` that is automatically trimmed to fit within a maximum
    number of tokens.
    """

    model_config = ConfigDict(frozen=True)
    tokenizer: Tokenizer
    max_tokens: int
    notions: List[Notion] = Field(default_factory=list)
    tokenized_notions: List[List[int]] = Field(default_factory=list)
    persistent_indices: set = Field(default_factory=set)

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_tokens: int,
        notions: List[Notion] = None,
        **kwargs,
    ):
        # Initialize with empty notions if None
        notions = notions or []

        # Initialize tokenized_notions
        tokenized_notions = [tokenizer.encode(notion.content) for notion in notions]

        # Call parent init with all values
        super().__init__(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            notions=notions,
            tokenized_notions=tokenized_notions,
            **kwargs,
        )

    @model_validator(mode="after")
    def validate_notions(cls, values):
        notions = values.notions
        for notion in notions:
            cls.validate_notion(notion, values.max_tokens, values.tokenizer)
        return values

    @classmethod
    def validate_notion(cls, notion: Notion, max_tokens: int, tokenizer: Tokenizer):
        if len(notion.content) == 0:
            raise ValueError("Notion content cannot be empty.")

        tokenized_notion = tokenizer.encode(notion.content)
        if len(tokenized_notion) > max_tokens:
            raise ValueError("Notion exceeds maximum token length")

        return tokenized_notion

    @property
    def total_tokens(self) -> int:
        """The total number of tokens in the Idearium."""
        return sum(len(notion) for notion in self.tokenized_notions)

    @property
    def _non_persistent_indices(self) -> set:
        """The indices of non-persistent notions."""
        return set(range(len(self.notions))) - self.persistent_indices

    def index(self, notion: Notion) -> int:
        """Returns the index of the first occurrence of the given notion."""
        return self.notions.index(notion)

    def append(self, notion: Notion):
        """Appends the given notion to the end of the Idearium."""
        logger.debug(f"Appending notion: {notion.content!r}")
        tokenized_notion = self.tokenizer.encode(notion.content)

        if self.notions:
            logger.debug(f"Current last notion: {self.notions[-1].content!r}")

        if (
            self.notions
            and self.notions[-1].role == notion.role
            and self.notions[-1].persistent == notion.persistent
        ):
            combined_content = self.notions[-1].content + notion.content
            combined_notion = Notion(
                content=combined_content,
                role=notion.role,
                persistent=notion.persistent,
            )
            self.replace(len(self.notions) - 1, combined_notion)
            logger.debug(
                f"After replace, about to return combined content: {combined_content!r}"
            )
            return

        logger.debug(f"Hitting append path. Appending new notion: {notion.content!r}")
        self.notions.append(notion)
        self.tokenized_notions.append(tokenized_notion)

        if notion.persistent:
            # Modify the set in place instead of reassigning
            self.persistent_indices.add(len(self.notions) - 1)

        self._trim()

    def extend(self, notions: Union[List[Notion], "Idearium"]):
        """Extends the Idearium with the given list of notions."""
        if isinstance(notions, Idearium):
            notions = notions.notions

        for notion in notions:
            self.append(notion)

    def insert(self, index: int, notion: Notion):
        """Inserts the given notion at the given index."""
        tokenized_notion = self.tokenizer.encode(notion.content)

        self.notions.insert(index, notion)
        self.tokenized_notions.insert(index, tokenized_notion)

        # Update persistent_indices in place
        new_indices = {i + 1 if i >= index else i for i in self.persistent_indices}
        self.persistent_indices.clear()
        self.persistent_indices.update(new_indices)
        if notion.persistent:
            self.persistent_indices.add(index)

        self._trim()

    def remove(self, notion: Notion):
        """Removes the first occurrence of the given notion."""
        index = self.index(notion)
        self.pop(index)

    def pop(self, index: int) -> Notion:
        """Removes and returns the notion at the given index."""
        ret = self.notions.pop(index)
        self.tokenized_notions.pop(index)

        # Update persistent_indices
        # Modify the set in place instead of reassigning
        self.persistent_indices.discard(index)
        new_indices = {i - 1 if i > index else i for i in self.persistent_indices}
        self.persistent_indices.clear()
        self.persistent_indices.update(new_indices)

        return ret

    def replace(self, index: int, notion: Notion):
        """Replaces the notion at the given index with the given notion."""
        self.notions[index] = notion
        self.tokenized_notions[index] = self.tokenizer.encode(notion.content)

        # Update persistent_indices based on the replaced notion
        if notion.persistent:
            self.persistent_indices.add(index)
        else:
            self.persistent_indices.discard(index)

        self._trim()

    def copy(self) -> "Idearium":
        """Returns a copy of the Idearium."""
        return Idearium(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            notions=self.notions.copy(),
            tokenized_notions=self.tokenized_notions.copy(),
            persistent_indices=self.persistent_indices.copy(),
        )

    def _trim(self):
        """
        Trims the Idearium to fit within the maximum number of tokens, called
        after every modification.

        This is the primary point of extension for Idearium subclasses, as it
        allows for custom trimming behavior.
        """
        while self.total_tokens > self.max_tokens:
            non_persistent_indices = self._non_persistent_indices

            # Check if there's only one non-persistent user message
            if len(non_persistent_indices) == 1:
                single_index = next(iter(non_persistent_indices))
                tokenized_notion = self.tokenized_notions[single_index]

                # Trim the only non-persistent notion to fit within the token limit
                tokenized_notion = tokenized_notion[
                    : self.max_tokens - (self.total_tokens - len(tokenized_notion))
                ]
                trimmed_content = self.tokenizer.decode(tokenized_notion)
                trimmed_notion = Notion(
                    content=trimmed_content,
                    role=self.notions[single_index].role,
                    persistent=self.notions[single_index].persistent,
                )
                self.replace(single_index, trimmed_notion)
                return

            # Attempt to remove the first non-persistent notion
            for i in non_persistent_indices:
                self.pop(i)
                break
            else:
                # If all notions are persistent and
                # the max token length is still exceeded
                raise ValueError(
                    "Persistent notions exceed max_tokens."
                    + " Reduce the content or increase max_tokens."
                )

    def __len__(self) -> int:
        return len(self.notions)

    def __getitem__(self, index: int) -> Notion:
        return self.notions[index]

    def __setitem__(self, index: int, notion: Notion):
        self.replace(index, notion)

    def __delitem__(self, index: int):
        self.pop(index)

    def __iter__(self) -> Iterator[Notion]:
        return iter(self.notions)

    def __contains__(self, notion: Notion) -> bool:
        return notion in self.notions

    def __str__(self) -> str:
        return str(self.notions)

    def __repr__(self) -> str:
        return repr(self.notions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Idearium):
            return NotImplemented
        return self.notions == other.notions
