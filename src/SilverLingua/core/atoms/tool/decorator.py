from functools import update_wrapper
from typing import Any, Callable

from .tool import Tool


class ToolWrapper:
    """A wrapper class that makes a function behave like a Tool instance."""

    def __init__(self, func: Callable):
        self._tool = Tool(function=func)
        update_wrapper(self, func)

        # Copy commonly accessed attributes
        self.function = self._tool.function
        self.description = self._tool.description
        self.name = self._tool.name
        self.use_function_call = self._tool.use_function_call

    def __call__(self, *args, **kwargs):
        return self._tool(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def __str__(self) -> str:
        return self._tool.model_dump_json()


def tool(func: Callable) -> ToolWrapper:
    """
    A decorator that converts a function into a Tool.
    This allows for a more concise way to create tools compared to using Tool(function).

    Example Usage:
    ```python
    @tool
    def add_numbers(x: int, y: int) -> int:
        '''Add two numbers together.'''
        return x + y

    # The function is now a Tool instance
    result = add_numbers(2, 3)  # Returns "5" (as a JSON string)
    ```
    """
    return ToolWrapper(func)
