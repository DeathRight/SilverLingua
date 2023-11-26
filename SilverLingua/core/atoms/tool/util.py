import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolCallResponse(BaseModel):
    """
    The response property of a tool call.
    """

    tool_call_id: str = Field(alias="id")
    name: str
    content: str = Field(alias="response")

    @classmethod
    def from_tool_call(cls, tool_call: "ToolCall", response: str) -> "ToolCallResponse":
        return cls(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            content=response,
        )


class ToolCallFunction(BaseModel):
    name: str = Field(default="")
    arguments: str = Field(default="")

    @field_validator("*", mode="before")
    def string_if_none(cls, v):
        return v if v is not None else ""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow", ignored_types=(type(None),))
    #
    function: ToolCallFunction
    id: str = Field(default_factory=lambda: str(uuid4()))
    index: Optional[int] = None

    @field_validator("id", mode="before")
    def string_if_none(cls, v):
        return v if v is not None else str(uuid4())

    def concat(self, other: "ToolCall") -> "ToolCall":
        """
        Concatenates two tool calls and returns the result.

        If the IDs are different, prioritize the ID of 'other'.
        For 'function', merge the 'name' and 'arguments' fields.

        We will prefer the `id` of self over other for 2 reasons:
        1. We assume that self is the older of the two
        2. The newer may be stream chunked, in which case
        the `id` of `other` may have been `None` and generated
        using UUID, but the older ID likely was generated
        by an API and thus this newer ID is not the true ID.
        """
        merged_function = {
            "name": (self.function.name or "") + (other.function.name or ""),
            "arguments": (self.function.arguments or "")
            + (other.function.arguments or ""),
        }
        merged_function = ToolCallFunction(**merged_function)

        self_extra = self.__pydantic_extra__
        other_extra = other.__pydantic_extra__

        # Compare the two extra fields
        if self_extra != other_extra:
            if not self_extra or not other_extra:
                return ToolCall(
                    id=self.id or other.id,
                    function=merged_function,
                    index=(self.index or other.index) or None,
                    **(self_extra or {}),
                    **(other_extra or {}),
                )
            # If they are different, merge them
            merged_extra = {}
            for key in set(self_extra.keys()) | set(other_extra.keys()):
                if key in self_extra and key in other_extra:
                    # If one is None, use the other
                    # Else, concatenate them
                    if self_extra[key] == other_extra[key]:
                        merged_extra[key] = self_extra[key]
                    else:
                        if self_extra[key] is None or other_extra[key] is None:
                            merged_extra[key] = self_extra[key] or other_extra[key]
                        else:
                            merged_extra[key] = self_extra[key] + other_extra[key]
                elif key in self_extra:
                    merged_extra[key] = self_extra[key]
                elif key in other_extra:
                    merged_extra[key] = other_extra[key]
            #
            return ToolCall(
                id=self.id or other.id,
                function=merged_function,
                index=(self.index or other.index) or None,
                **merged_extra,
            )

        return ToolCall(
            id=self.id or other.id,
            function=merged_function,
            index=(self.index or other.index) or None,
            **(self_extra or {}),
        )


class ToolCalls(BaseModel):
    """
    A list of tool calls.
    """

    list: List[ToolCall] = Field(default_factory=list, frozen=True)

    def concat(self, other: "ToolCalls") -> "ToolCalls":
        """
        Concatenates two tool calls lists and returns the result.
        """
        new: List[ToolCall] = self.list.copy()
        for tool_call in other.list:
            found = False
            # Find the tool call with the same ID
            for i, self_tool_call in enumerate(new):
                if (
                    self_tool_call.id == tool_call.id
                    or self_tool_call.index == tool_call.index
                ):
                    new[i] = self_tool_call.concat(tool_call)
                    found = True
            if not found:
                new.append(tool_call)
        return ToolCalls(list=new)


class Parameter(BaseModel):
    """
    The parameter of a function according to JSON schema standards.
    (Used by OpenAI function calling)
    """

    type: str
    description: Optional[str] = None
    enum: Optional[list[str]] = None


class Parameters(BaseModel):
    """
    The parameters property of a function according to
    JSON schema standards. (Used by OpenAI function calling)
    """

    type: str
    properties: Dict[str, Parameter] = {}
    required: Optional[List[str]] = None


class FunctionJSONSchema(BaseModel):
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
            sides: The number of sides on each die (default 20)
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

        capturing_description = True
        start_capturing = False
        for line in args_lines:
            if line.strip().lower() in ["args:", "arguments:"]:
                start_capturing = True
                capturing_description = False
                continue

            if capturing_description:
                description += "\n" + line
            elif start_capturing:
                match = re.match(r"^\s+(?P<name>\w+):\s(?P<desc>.*)", line)
                if match:
                    args_docs[match.group("name")] = match.group("desc")

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        param_type = param.annotation if param.annotation is not inspect._empty else Any
        type_schema = python_type_to_json_schema_type(param_type)

        parameter_info = {"description": args_docs.get(name, "")}

        # Handle different types of type_schema
        if isinstance(type_schema, str):
            parameter_info["type"] = type_schema
        elif isinstance(type_schema, dict):
            parameter_info.update(type_schema)

        properties[name] = Parameter(**parameter_info)

        if param.default is param.empty:
            required.append(name)

    parameters_model = Parameters(
        type="object", properties=properties, required=required or None
    )

    return FunctionJSONSchema(
        name=func.__name__, description=description.strip(), parameters=parameters_model
    )
