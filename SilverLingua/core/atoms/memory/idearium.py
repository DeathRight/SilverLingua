# A collection of Notions (should be possible to set up for automatic trimming of
# notions or their content, with awareness of roles)

from typing import Callable, List, TypedDict

from .notion import Notion


class Tokenizer(TypedDict):
    encode: Callable[[str], List[int]]
    decode: Callable[[List[int]], str]


class Idearium:
    def __init__(
        self, tokenizer: Tokenizer, max_tokens: int, notions: List[Notion] = None
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.notions = []
        self.tokenized_notions = []
        self.persistent_indices = (
            set()
        )  # Keeping track of indices with persistent notions

        if notions is not None:
            for notion in notions:
                self.add(notion)

    @property
    def total_tokens(self) -> int:
        return sum(len(notion) for notion in self.tokenized_notions)

    @property
    def _non_persistent_indices(self) -> set:
        return set(range(len(self.notions))) - self.persistent_indices

    def index(self, notion: Notion) -> int:
        return self.notions.index(notion)

    def add(self, notion: Notion):
        tokenized_notion = self.tokenizer["encode"](notion.content)

        if len(tokenized_notion) > self.max_tokens:
            raise ValueError("Notion exceeds maximum token length")

        self.notions.append(notion)
        self.tokenized_notions.append(tokenized_notion)

        if notion.persistent:
            self.persistent_indices.add(len(self.notions) - 1)

    def remove(self, notion: Notion):
        index = self.index(notion)
        self.pop(index)

    def pop(self, index: int) -> Notion:
        ret = self.notions.pop(index)
        self.tokenized_notions.pop(index)

        # Update persistent_indices
        self.persistent_indices.discard(index)
        self.persistent_indices = {
            i - 1 if i > index else i for i in self.persistent_indices
        }

        return ret

    def replace(self, index: int, notion: Notion):
        self.notions[index] = notion
        self.tokenized_notions[index] = self.tokenizer["encode"](notion.content)

        # Update persistent_indices based on the replaced notion
        if notion.persistent:
            self.persistent_indices.add(index)
        else:
            self.persistent_indices.discard(index)

    def _trim(self):
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
                self.notions[single_index].content = self.tokenizer["decode"](
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
