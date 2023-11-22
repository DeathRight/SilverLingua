from typing import Callable, Iterator, List, TypedDict

from .notion import Notion


class Tokenizer(TypedDict):
    """
    A tokenizer that can encode and decode strings.
    """

    encode: Callable[[str], List[int]]
    decode: Callable[[List[int]], str]


class Idearium:
    """
    A collection of `Notions` that is automatically trimmed to fit within a maximum
    number of tokens.
    """

    def __init__(
        self, tokenizer: Tokenizer, max_tokens: int, notions: List[Notion] = None
    ):
        """
        Args:
            tokenizer: A tokenizer that can encode and decode strings.
            max_tokens: The maximum number of tokens the Idearium can contain.
            notions: A list of notions to initialize the Idearium with. (Optional)
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._notions = []
        self.tokenized_notions = []
        self.persistent_indices = (
            set()
        )  # Keeping track of indices with persistent notions

        if notions is not None:
            for notion in notions:
                self.add(notion)

    @property
    def total_tokens(self) -> int:
        """The total number of tokens in the Idearium."""
        return sum(len(notion) for notion in self.tokenized_notions)

    @property
    def _non_persistent_indices(self) -> set:
        """The indices of non-persistent notions."""
        return set(range(len(self._notions))) - self.persistent_indices

    def index(self, notion: Notion) -> int:
        """Returns the index of the first occurrence of the given notion."""
        return self._notions.index(notion)

    def append(self, notion: Notion):
        """Appends the given notion to the end of the Idearium."""
        tokenized_notion = self.tokenizer.encode(notion.content)

        if len(tokenized_notion) > self.max_tokens:
            raise ValueError("Notion exceeds maximum token length")

        self._notions.append(notion)
        self.tokenized_notions.append(tokenized_notion)

        if notion.persistent:
            self.persistent_indices.add(len(self._notions) - 1)

        self._trim()

    def extend(self, notions: List[Notion]):
        """Extends the Idearium with the given list of notions."""
        for notion in notions:
            self.append(notion)

    def insert(self, index: int, notion: Notion):
        """Inserts the given notion at the given index."""
        tokenized_notion = self.tokenizer["encode"](notion.content)

        if len(tokenized_notion) > self.max_tokens:
            raise ValueError("Notion exceeds maximum token length")

        self._notions.insert(index, notion)
        self.tokenized_notions.insert(index, tokenized_notion)

        # Update persistent_indices
        self.persistent_indices = {
            i + 1 if i >= index else i for i in self.persistent_indices
        }
        if notion.persistent:
            self.persistent_indices.add(index)

        self._trim()

    def remove(self, notion: Notion):
        """Removes the first occurrence of the given notion."""
        index = self.index(notion)
        self.pop(index)

    def pop(self, index: int) -> Notion:
        """Removes and returns the notion at the given index."""
        ret = self._notions.pop(index)
        self.tokenized_notions.pop(index)

        # Update persistent_indices
        self.persistent_indices.discard(index)
        self.persistent_indices = {
            i - 1 if i > index else i for i in self.persistent_indices
        }

        return ret

    def replace(self, index: int, notion: Notion):
        """Replaces the notion at the given index with the given notion."""
        self._notions[index] = notion
        self.tokenized_notions[index] = self.tokenizer["encode"](notion.content)

        # Update persistent_indices based on the replaced notion
        if notion.persistent:
            self.persistent_indices.add(index)
        else:
            self.persistent_indices.discard(index)

        self._trim()

    def copy(self) -> "Idearium":
        """Returns a copy of the Idearium."""
        return Idearium(self.tokenizer, self.max_tokens, self._notions.copy())

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
                self._notions[single_index].content = self.tokenizer["decode"](
                    tokenized_notion
                )
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
        return len(self._notions)

    def __getitem__(self, index: int) -> Notion:
        return self._notions[index]

    def __setitem__(self, index: int, notion: Notion):
        self.replace(index, notion)

    def __delitem__(self, index: int):
        self.pop(index)

    def __iter__(self) -> Iterator[Notion]:
        return iter(self._notions)

    def __contains__(self, notion: Notion) -> bool:
        return notion in self._notions

    def __str__(self) -> str:
        return str(self._notions)

    def __repr__(self) -> str:
        return repr(self._notions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Idearium):
            return NotImplemented
        return self._notions == other._notions
