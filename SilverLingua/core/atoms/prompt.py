from functools import wraps
from typing import Callable

from jinja2 import Template


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
        template = Template(docstring)

        # Bind arguments to parameter names
        # Filter out 'return' from the annotations' keys
        bound_args = [key for key in func.__annotations__ if key != "return"]

        arg_dict = {}
        for name, value in zip(bound_args, args, strict=True):
            arg_dict[name] = value
        arg_dict.update(kwargs)

        # Render the template
        rendered = template.render(**arg_dict)

        # Strip each line and remove leading/trailing whitespaces
        stripped_lines = [line.lstrip() for line in rendered.splitlines()]
        return "\n".join(stripped_lines).strip()

    return wrapper


@prompt
def RolePrompt(role: str, text: str):  # type: ignore
    """{role}: {text}"""
