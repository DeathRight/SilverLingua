from typing import List, Literal, Optional

from pydantic import BaseModel, Field

AnthropicModelName = Literal[
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]

AnthropicModels = {
    "claude-3-opus-20240229": {
        "max_tokens": 4096,
        "input_cost": 0.015,
        "output_cost": 0.075,
    },
    "claude-3-sonnet-20240229": {
        "max_tokens": 4096,
        "input_cost": 0.003,
        "output_cost": 0.015,
    },
    "claude-3-haiku-20240307": {
        "max_tokens": 4096,
        "input_cost": 0.00025,
        "output_cost": 0.00125,
    },
    "claude-2.1": {"max_tokens": 4096, "input_cost": 0.008, "output_cost": 0.024},
    "claude-2.0": {"max_tokens": 4096, "input_cost": 0.008, "output_cost": 0.024},
    "claude-instant-1.2": {
        "max_tokens": 4096,
        "input_cost": 0.00163,
        "output_cost": 0.00551,
    },
}


class CompletionParams(BaseModel):
    """Parameters for completion requests."""

    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    tools: Optional[List[dict]] = Field(default=None)
    stream: Optional[bool] = Field(default=None)
