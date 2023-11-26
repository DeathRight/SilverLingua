import json
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from .util import (
    FunctionJSONSchema,
    ToolCallFunction,
    generate_function_json,
)


class Tool(BaseModel):
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

    Alternatively, you can call the function using a ToolCallFunction object.
        ```python
        # Create a FunctionCall object
        function_call = FunctionCall("my_function", {"x": 1, "y": 2})

        # Call the function using the FunctionCall object
        result = tool_instance(function_call)  # Output will be 3
        ```
    """

    function: Callable
    description: FunctionJSONSchema = Field(validate_default=True)
    name: str = Field(validate_default=True)

    @field_validator("description", mode="before")
    def set_default_description(cls, v, values):
        return generate_function_json(values["function"])

    @field_validator("name", mode="before")
    def set_default_name(cls, v, values):
        return values["description"]["name"]

    def use_function_call(self, function_call: ToolCallFunction):
        """
        Uses a FunctionCall to call the function.
        """
        arguments_dict = function_call.arguments
        if arguments_dict == "":
            return json.dumps(self.function())

        try:
            arguments_dict = json.loads(function_call.arguments)
        except json.JSONDecodeError:
            raise ValueError(
                "ToolCall.arguments must be a JSON string.\n"
                + f"function_call.arguments: {function_call.arguments}\n"
                + f"json.loads result: {arguments_dict}"
            ) from None

        return json.dumps(self.function(**arguments_dict))

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], ToolCallFunction):
            return self.use_function_call(args[0])
        return json.dumps(self.function(*args, **kwargs))

    def __str__(self):
        return self.to_json()
