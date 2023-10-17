import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, TypedDict, Union


@dataclass(frozen=True)
class FunctionCall:
    """
    The JSON output of a function call.

    When the AI attempts to call a function, this is the schema.

    [Note: The str representation is a JSON string.]
    """

    name: str
    arguments: Dict[str, Any]

    @classmethod
    def from_json(cls, json_str: str) -> "FunctionCall":
        data = json.loads(json_str)
        return cls(name=data["name"], arguments=data["arguments"])

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    def __repr__(self) -> str:
        return self.to_json()

    def __str__(self) -> str:
        return self.to_json()


@dataclass(frozen=True)
class FunctionResponse:
    """
    The JSON output of a function response.

    When the system responds to a function call by the AI, this is the schema.

    Content must be a string in JSON format.

    [Note: The str representation is a JSON string.]
    """

    name: str
    content: str

    @classmethod
    def from_json(cls, json_str: str) -> "FunctionResponse":
        data = json.loads(json_str)
        return cls(name=data["name"], response=data["content"])

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    def __repr__(self) -> str:
        return self.to_json()

    def __str__(self) -> str:
        return self.to_json()


class Parameter(TypedDict, total=False):
    """
    The parameter of a function according to JSON schema standards.
    (Used by OpenAI function calling)
    """

    type: str
    description: Optional[str]
    enum: Optional[list[str]]


class Parameters(TypedDict, total=False):
    """
    The parameters property of a function according to
    JSON schema standards. (Used by OpenAI function calling)
    """

    type: str
    properties: dict[str, Parameter]
    required: list[str]


class FunctionJSONSchema(TypedDict):
    """
    A function according to JSON schema standards.

    This is also passed in to OpenAI ChatCompletion API
    functions list so the AI understands how to call a function.

    Attributes:
        name: The name of the function
        description: A description of the function
        parameters: A dictionary of parameters and their types (Optional)

    Example:

    ```json
    {
        "name": "roll_dice",
        "description": "Rolls a number of dice with a given number of sides, optionally
            with a modifier and/or advantage/disadvantage.
            Returns `{result: int, rolls: int[]}`",
        "parameters": {
            "type": "object",
            "properties": {
                "sides": {
                  "description": "The number of sides on each die",
                  "type": "integer"
                },
                "dice": {
                  "description": "The number of dice to roll (default 1)",
                  "type": "integer"
                },
                "modifier": {
                  "description": "The modifier to add to the roll total (default 0)",
                  "type": "integer"
                },
                "advantage": {
                  "description": "Whether to roll with advantage (default False)",
                  "type": "boolean"
                },
                "disadvantage": {
                  "description": "Whether to roll with disadvantage (default False)",
                  "type": "boolean"
                }
            }
        }
    }
    ```
    """

    name: str
    description: str
    parameters: Parameters


##########################################################################################
# Global Functions
##########################################################################################
def python_type_to_json_schema_type(python_type: Type[Any]) -> Union[str, Dict]:
    """
    Maps Python types to JSON Schema types.

    Args:
      python_type: The Python type.

    Returns:
      Union[str, Dict]: The corresponding JSON Schema type or schema.
    """
    simple_type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        type(None): "null",
    }

    if python_type in simple_type_mapping:
        return simple_type_mapping[python_type]

    if hasattr(python_type, "__origin__"):
        origin = python_type.__origin__  # type: ignore

        if origin is Union:
            types = python_type.__args__  # type: ignore
            if type(None) in types:
                # This is equivalent to Optional[T]
                types = [t for t in types if t is not type(None)]
                if len(types) == 1:
                    return python_type_to_json_schema_type(types[0])

        if origin is list:
            item_type = (
                python_type.__args__[0] if python_type.__args__ else Any
            )  # type: ignore
            return {
                "type": "array",
                "items": {"type": python_type_to_json_schema_type(item_type)},
            }
        elif origin is dict:
            key_type = (
                python_type.__args__[0] if python_type.__args__ else Any
            )  # type: ignore
            value_type = (
                python_type.__args__[1] if python_type.__args__ else Any
            )  # type: ignore
            if key_type is not str:
                raise ValueError(
                    "Dictionary key type must be str for conversion to JSON schema"
                )
            return {
                "type": "object",
                "additionalProperties": python_type_to_json_schema_type(value_type),
            }

    if hasattr(python_type, "__annotations__"):
        properties = {}
        for k, v in python_type.__annotations__.items():
            type_schema = python_type_to_json_schema_type(v)
            if isinstance(type_schema, str):
                properties[k] = {"type": type_schema}
            elif isinstance(type_schema, dict):
                properties[k] = type_schema
        required = [
            k
            for k in properties
            if k not in python_type.__optional_keys__  # type: ignore
        ]
        return {
            "type": "object",
            "properties": properties,
            **({"required": required} if required else {}),
        }

    print(f"Unknown type encountered: {python_type}")
    return "unknown"


def generate_function_json(func: Callable[..., Any]) -> FunctionJSONSchema:
    """
    Generates a FunctionJSONSchema from a python function.

    Example:
    ```python
    def roll_dice(sides: int = 20,
                  dice: int = 1,
                  modifier: int = 0,
                  advantage: bool = False,
                  disadvantage: bool = False):
        \"""
        Rolls a number of dice with a given number of sides, optionally with a modifier
        and/or advantage/disadvantage.
        Returns `{result: int, rolls: int[]}`

        Args:
            sides: The number of sides on each die
            dice: The number of dice to roll (default 1)
            modifier: The modifier to add to the roll total (default 0)
            advantage: Whether to roll with advantage (default False)
            disadvantage: Whether to roll with disadvantage (default False)
        \"""
        ...
    ```

    Usage:
    ```python
    result = generate_function_json(roll_dice)
    print(result)
    ```

    Expected Output:
    ```json
    {
        "name": "roll_dice",
        "description": "Rolls a number of dice with a given number of sides, optionally
            with a modifier and/or advantage/disadvantage.
            Returns `{result: int, rolls: int[]}`",
        "parameters": {
            "type": "object",
            "properties": {
                "sides": {
                  "description": "The number of sides on each die",
                  "type": "integer"
                },
                "dice": {
                  "description": "The number of dice to roll (default 1)",
                  "type": "integer"
                },
                "modifier": {
                  "description": "The modifier to add to the roll total (default 0)",
                  "type": "integer"
                },
                "advantage": {
                  "description": "Whether to roll with advantage (default False)",
                  "type": "boolean"
                },
                "disadvantage": {
                  "description": "Whether to roll with disadvantage (default False)",
                  "type": "boolean"
                }
            }
        }
    }
    ```
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func)

    description = ""
    args_docs = {}
    if doc:
        doc_lines = doc.split("\n")
        description = doc_lines[0]  # Capture the first line as part of the description.
        args_lines = doc_lines[1:]

        # Initialize a flag for capturing the description
        capturing_description = True

        # Look for the "Args" or "Arguments" keyword before starting to capture
        start_capturing = False
        for line in args_lines:
            if line.strip().lower() in ["args:", "arguments:"]:
                start_capturing = True
                capturing_description = False
                continue

            if capturing_description:
                # Keep appending to the description
                description += "\n" + line
            elif start_capturing:
                # Capture the argument name and description
                match = re.match(r"^\s+(?P<name>\w+):\s(?P<desc>.*)", line)
                if match:
                    args_docs[match.group("name")] = {
                        "description": match.group("desc")
                    }

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        param_type = param.annotation if param.annotation is not inspect._empty else Any
        type_schema = python_type_to_json_schema_type(param_type)

        properties[name] = {
            "description": args_docs.get(name, {}).get("description", None)
        }
        if isinstance(type_schema, str):
            properties[name]["type"] = type_schema
        elif isinstance(type_schema, dict):
            properties[name].update(type_schema)

        if param.default is param.empty:
            required.append(name)

    parameters: Parameters = {"type": "object", "properties": properties}

    if required:
        parameters["required"] = required

    return {
        "name": func.__name__,
        "description": description.strip(),
        "parameters": parameters,
    }
