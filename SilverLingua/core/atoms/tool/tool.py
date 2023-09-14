import json
from typing import Callable

from .util import FunctionJSONSchema, generate_function_json


class Tool:
  """
    A wrapper class for functions that allows them to be both directly callable
    and serializable to JSON for use with an LLM.

    Attributes:
        function (Callable): The function to be wrapped.
        description (FunctionJSONSchema): A TypedDict that describes the function
            according to JSON schema standards.
        name (str): The name of the function, extracted from the FunctionJSONSchema.

    Example:
        ```python
        def my_function(x, y):
            return x + y

        # Create a Tool instance
        tool_instance = Tool(my_function)

        # Directly call the wrapped function
        result = tool_instance(1, 2)  # Output will be 3

        # Serialize to JSON
        serialized = str(tool_instance)
        ```
    """

  function: Callable
  description: FunctionJSONSchema
  name: str

  def __init__(self, function: Callable):
    self.function = function
    self.description = generate_function_json(function)
    self.name = self.description['name']

  def __call__(self, *args, **kwargs):
    return self.function(*args, **kwargs)

  def to_json(self):
    return json.dumps(self.description)

  def __str__(self):
    return self.to_json()
