from functools import wraps
from inspect import signature
from typing import Callable

from jinja2 import StrictUndefined, Template


def prompt(func: Callable) -> Callable[..., str]:
    """
    A decorator to render a function's docstring as a Jinja2 template.
    Uses the function arguments as variables for the template.

    Note: Be deliberate about new lines in your docstrings - they
    may make meaningful changes in an AI's output. For instance,
    separating long sentences with a newline for human
    readability may cause issues.

    Example Usage:
    ```python
    @prompt
    def fruit_prompt(fruits: list) -> None:
        \"""
        You are a helpful assistant that takes a list of fruit and gives information about their nutrition.

        LIST OF FRUIT:
        {% for fruit in fruits %}{{ fruit }}
        {% endfor %}
        \"""

    print(fruit_prompt(["apple", "orange"]))
    ```

    Expected Output:
    ```
    You are a helpful assistant that takes a list of fruit and gives information about their nutrition.

    LIST OF FRUIT:
    apple
    orange
    ```
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        docstring = func.__doc__ or ""
        template = Template(docstring, undefined=StrictUndefined)

        # Get function signature and bind arguments
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Render the template with bound arguments
        rendered = template.render(**bound_args.arguments)

        # Strip each line and remove leading/trailing whitespaces
        stripped_lines = [line.lstrip() for line in rendered.splitlines()]
        return "\n".join(stripped_lines).strip()

    return wrapper


@prompt
def RolePrompt(role: str, text: str):  # type: ignore
    """{{ role }}: {{ text }}"""
