from typing import Generator, List, Union

import pytest

from silverlingua.core.atoms import ChatRole, Tokenizer
from silverlingua.core.molecules import Notion
from silverlingua.core.templates.model import Messages, Model, ModelType


class MockTokenizer(Tokenizer):
    """Mock tokenizer for testing."""

    def __init__(self):
        def encode(text: str) -> list[int]:
            return list(range(len(text)))  # Each character is a token

        def decode(tokens: list[int]) -> str:
            return "x" * len(tokens)  # Return string of length equal to tokens

        super().__init__(encode=encode, decode=decode)


class MockModel(Model):
    """Mock model implementation for testing."""

    completion_params: dict = {}  # Add as a field

    def __init__(self, max_response: int = 100):
        def mock_llm(**kwargs):
            """Mock synchronous LLM call."""
            return {"response": "This is a mock response"}

        async def mock_llm_async(**kwargs):
            """Mock asynchronous LLM call."""
            return {"response": "This is a mock async response"}

        super().__init__(
            max_response=max_response,
            api_key="mock-api-key",
            name="mock-model",
            role=ChatRole,
            type=ModelType.CHAT,
            llm=mock_llm,
            llm_async=mock_llm_async,
            can_stream=True,
            tokenizer=MockTokenizer(),
        )

    @property
    def max_tokens(self) -> int:
        return 100

    def _format_request(self, messages: List[Notion], *args, **kwargs) -> dict:
        """Format messages into a mock request."""
        return {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }

    def _standardize_response(
        self, response: Union[dict, str, List[any]], *args, **kwargs
    ) -> List[Notion]:
        """Convert mock response to notions."""
        if isinstance(response, dict):
            return [Notion(content=response["response"], role=self.role.AI)]
        return [Notion(content=str(response), role=self.role.AI)]

    def _postprocess(self, response: List[Notion], *args, **kwargs) -> List[Notion]:
        """No post-processing needed for mock."""
        return response

    def _retry_call(
        self,
        input: Union[str, dict, List[any]],
        e: Exception,
        api_call: callable,
        retries: int = 0,
    ) -> Union[str, dict]:
        """Mock retry logic."""
        return {"response": "This is a retry response"}

    def generate(self, messages: Messages, **kwargs) -> List[Notion]:
        """Synchronous generation."""
        return self._common_generate_logic(messages, is_async=False, **kwargs)

    async def agenerate(self, messages: Messages, **kwargs) -> List[Notion]:
        """Asynchronous generation."""
        return await self._common_generate_logic(messages, is_async=True, **kwargs)

    def stream(self, messages: Messages, **kwargs) -> Generator[Notion, None, None]:
        """Synchronous streaming."""
        response = self.generate(messages, **kwargs)
        for notion in response:
            yield notion

    async def astream(
        self, messages: Messages, **kwargs
    ) -> Generator[Notion, None, None]:
        """Asynchronous streaming."""
        response = await self.agenerate(messages, **kwargs)
        for notion in response:
            yield notion


@pytest.fixture
def model():
    return MockModel()


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.model
@pytest.mark.unit
def test_model_initialization(model):
    """Test basic model initialization."""
    assert model.api_key == "mock-api-key"
    assert model.name == "mock-model"
    assert model.type == ModelType.CHAT
    assert model.can_stream is True
    assert model.max_tokens == 100


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.model
@pytest.mark.unit
def test_process_input_string(model):
    """Test processing string input."""
    result = model._process_input("Hello")
    assert len(result) == 1
    assert result[0].content == "Hello"
    assert result[0].role == str(ChatRole.HUMAN.value)


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.model
@pytest.mark.unit
def test_process_input_notion(model):
    """Test processing Notion input."""
    notion = Notion(content="Hello", role=ChatRole.HUMAN)
    result = model._process_input(notion)
    assert len(result) == 1
    assert result[0] == notion


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.model
@pytest.mark.unit
def test_process_input_list(model):
    """Test processing list input."""
    inputs = ["Hello", Notion(content="World", role=ChatRole.AI)]
    result = model._process_input(inputs)
    assert len(result) == 2
    assert result[0].content == "Hello"
    assert result[0].role == str(ChatRole.HUMAN.value)
    assert result[1].content == "World"
    assert result[1].role == str(ChatRole.AI.value)


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.model
@pytest.mark.unit
def test_model_lifecycle(model):
    """Test the complete model lifecycle."""
    # Input processing
    messages = model._process_input("Hello")
    assert len(messages) == 1

    # Preprocessing
    preprocessed = model._preprocess(messages.notions)
    assert len(preprocessed) == 1
    assert preprocessed[0].content == "Hello"

    # Format request
    request = model._format_request(preprocessed)
    assert isinstance(request, dict)
    assert "messages" in request

    # Model call
    response = model._call(request)
    assert isinstance(response, dict)
    assert "response" in response

    # Standardize response
    standardized = model._standardize_response(response)
    assert isinstance(standardized, list)
    assert len(standardized) == 1
    assert isinstance(standardized[0], Notion)

    # Post-processing
    final = model._postprocess(standardized)
    assert isinstance(final, list)
    assert len(final) == 1
    assert isinstance(final[0], Notion)


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.model
@pytest.mark.unit
def test_model_retry_logic(model):
    """Test model retry logic."""

    def failing_api_call(**kwargs):
        raise Exception("API Error")

    # Should get retry response after failure
    response = model._common_call_logic({"test": "input"}, failing_api_call, retries=0)
    assert response == {"response": "This is a retry response"}

    # Should raise after max retries
    with pytest.raises(Exception):
        model._common_call_logic({"test": "input"}, failing_api_call, retries=3)


def test_model_role_conversion(model):
    """Test role conversion."""
    human_role = model._convert_role(ChatRole.HUMAN)
    ai_role = model._convert_role(ChatRole.AI)
    system_role = model._convert_role(ChatRole.SYSTEM)

    assert human_role == str(ChatRole.HUMAN.value)
    assert ai_role == str(ChatRole.AI.value)
    assert system_role == str(ChatRole.SYSTEM.value)


@pytest.mark.asyncio
async def test_model_async_call(model):
    """Test asynchronous model call."""
    request = model._format_request([Notion(content="Hello", role=ChatRole.HUMAN)])
    response = await model._acall(request)
    assert isinstance(response, dict)
    assert response["response"] == "This is a mock async response"
