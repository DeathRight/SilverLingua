import json
import os
from typing import List

import pytest
from dotenv import load_dotenv

from silverlingua.core.atoms.tool import Tool
from silverlingua_anthropic import AsyncAnthropic
from silverlingua_openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def sample_tool(x: int) -> int:
    """A simple tool that doubles a number.

    Args:
        x: The number to double
    """
    return x * 2


def another_tool(text: str) -> str:
    """A simple tool that reverses text.

    Args:
        text: The text to reverse
    """
    return text[::-1]


@pytest.fixture
def tools() -> List[Tool]:
    """Fixture providing test tools."""
    return [
        Tool(function=sample_tool),
        Tool(function=another_tool),
    ]


@pytest.fixture
async def openai_client() -> AsyncOpenAI:
    """Fixture providing OpenAI client."""
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")
    return AsyncOpenAI(api_key=OPENAI_API_KEY)


@pytest.fixture
async def anthropic_client() -> AsyncAnthropic:
    """Fixture providing Anthropic client."""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_openai_streaming_parallel_tools(
    openai_client: AsyncOpenAI, tools: List[Tool]
):
    """Test OpenAI streaming with multiple parallel tool calls."""
    print("\nSetting up parallel tools test...")

    functions = [
        {
            "type": "function",
            "function": {
                "name": tool.description.name,
                "description": tool.description.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": param.type,
                            "description": param.description or "",
                        }
                        for name, param in tool.description.parameters.properties.items()
                    },
                    "required": tool.description.parameters.required or [],
                },
            },
        }
        for tool in tools
    ]
    print(f"\nTools configured: {json.dumps(functions, indent=2)}")

    print("\nStarting stream...")
    stream = await openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": "First double 42, then reverse the text 'hello world'.",
            }
        ],
        tools=functions,
        stream=True,
    )
    print("\nStream created, waiting for chunks...")

    chunks = []
    async for chunk in stream:
        print(f"\nReceived chunk: {chunk}")
        if chunk.choices[0].delta.tool_calls:
            tool_call = chunk.choices[0].delta.tool_calls[0]
            chunks.append(
                {
                    "index": tool_call.index,
                    "id": tool_call.id,
                    "function": (
                        {
                            "name": (
                                tool_call.function.name
                                if tool_call.function.name
                                else None
                            ),
                            "arguments": (
                                tool_call.function.arguments
                                if tool_call.function.arguments
                                else None
                            ),
                        }
                        if tool_call.function
                        else None
                    ),
                    "type": tool_call.type,
                }
            )
            print(f"Tool call chunk received: {tool_call}")

    print("\nFull parallel tool calls:", json.dumps(chunks, indent=2))
