import asyncio
import os

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def anthropic_client():
    """Fixture providing Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    try:
        from silverlingua_anthropic import Anthropic

        return Anthropic(api_key=api_key)
    except ImportError:
        pytest.skip("Anthropic package not installed")


@pytest.fixture
async def async_anthropic_client():
    """Fixture providing Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    try:
        from silverlingua_anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=api_key)
    except ImportError:
        pytest.skip("Anthropic package not installed")


@pytest.mark.anthropic
def test_standard_completion(anthropic_client):
    """Test standard completion without streaming or tool calls"""
    response = anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    assert response.content[0].text is not None
    assert "Paris" in response.content[0].text


@pytest.mark.anthropic
def test_standard_completion_with_tool(anthropic_client):
    """Test standard completion with tool calls"""
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    response = anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What's the weather like in New York?"}],
        tools=tools,
    )

    # Check if tool calls are present in the response
    assert response.content[0].text is not None
    print(f"Tool response: {response.content[0].text}")
    # Note: We'll need to verify the exact structure of tool calls in Anthropic's response


@pytest.mark.anthropic
async def test_streaming_completion_with_tool(async_anthropic_client):
    """Test streaming completion with tool calls"""
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    stream = await async_anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        tools=tools,
        stream=True,
    )

    collected_content = []
    tool_calls = []

    async for chunk in stream:
        print(f"Chunk type: {chunk.type}")
        print(f"Chunk content: {chunk}")

        if chunk.type == "content_block_delta":
            if hasattr(chunk.delta, "text"):
                collected_content.append(chunk.delta.text)
        elif chunk.type == "input_json_delta":
            tool_calls.append(chunk.delta.partial_json)

    # Verify both content and tool calls are present
    full_response = "".join(collected_content)
    print(f"Streaming tool response: {full_response}")
    print(f"Tool calls: {tool_calls}")

    # At least one of them should have content
    assert len(collected_content) > 0 or len(tool_calls) > 0


@pytest.mark.anthropic
async def test_parallel_completions(async_anthropic_client):
    """Test parallel completion requests"""

    async def make_request(question: str) -> str:
        response = await async_anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text

    questions = [
        "What is 2+2?",
        "What is the capital of Japan?",
        "What color is the sky?",
    ]

    tasks = [make_request(q) for q in questions]
    responses = await asyncio.gather(*tasks)

    print(f"Parallel responses: {responses}")
    assert len(responses) == 3
    assert any("4" in r for r in responses)
    assert any("Tokyo" in r for r in responses)
    assert any("blue" in r.lower() for r in responses)


if __name__ == "__main__":
    pytest.main([__file__])
