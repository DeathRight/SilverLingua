import pytest

from silverlingua.core.atoms.prompt import RolePrompt, prompt


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_basic_prompt():
    """Test basic prompt functionality."""

    @prompt
    def test_prompt(name: str):
        """Hello {{ name }}!"""

    result = test_prompt("World")
    assert result == "Hello World!"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_with_multiple_args():
    """Test prompt with multiple arguments."""

    @prompt
    def test_prompt(name: str, age: int):
        """Name: {{ name }}, Age: {{ age }}"""

    result = test_prompt("Alice", 30)
    assert result == "Name: Alice, Age: 30"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_with_kwargs():
    """Test prompt with keyword arguments."""

    @prompt
    def test_prompt(name: str, age: int):
        """Name: {{ name }}, Age: {{ age }}"""

    result = test_prompt(name="Bob", age=25)
    assert result == "Name: Bob, Age: 25"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_with_loops():
    """Test prompt with Jinja2 loops."""

    @prompt
    def test_prompt(items: list):
        """Items:{% for item in items %}
        - {{ item }}{% endfor %}"""

    result = test_prompt(["apple", "banana", "orange"])
    expected = "Items:\n- apple\n- banana\n- orange"
    assert result == expected


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_with_conditionals():
    """Test prompt with Jinja2 conditionals."""

    @prompt
    def test_prompt(name: str, age: int):
        """{{ name }} is {% if age >= 18 %}an adult{% else %}a minor{% endif %}."""

    assert test_prompt("Alice", 20) == "Alice is an adult."
    assert test_prompt("Bob", 15) == "Bob is a minor."


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_with_filters():
    """Test prompt with Jinja2 filters."""

    @prompt
    def test_prompt(text: str):
        """{{ text | upper }}"""

    result = test_prompt("hello")
    assert result == "HELLO"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_whitespace_handling():
    """Test prompt whitespace handling."""

    @prompt
    def test_prompt(items: list):
        """Start{% for item in items %}
        {{ item }}{% endfor %}
        End"""

    result = test_prompt(["a", "b"])
    expected = "Start\na\nb\nEnd"
    assert result == expected


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_empty_docstring():
    """Test prompt with empty docstring."""

    @prompt
    def test_prompt(name: str):
        pass  # No docstring

    result = test_prompt("test")
    assert result == ""


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_role_prompt():
    """Test the built-in RolePrompt."""
    result = RolePrompt("Assistant", "Hello, how can I help?")
    assert result == "Assistant: Hello, how can I help?"


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_error_cases():
    """Test error cases."""

    # Test missing required argument
    @prompt
    def test_prompt(name: str):
        """Hello {{ name }}!"""

    with pytest.raises(TypeError):
        test_prompt()  # Missing required argument

    # Test undefined variable in template
    @prompt
    def test_prompt(name: str):
        """Hello {{ undefined_var }}!"""

    with pytest.raises(Exception):  # Jinja2 will raise an exception
        test_prompt("test")


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.prompt
@pytest.mark.unit
def test_prompt_type_annotations():
    """Test that type annotations are properly handled."""

    @prompt
    def test_prompt(name: str, count: int, items: list[str]) -> str:
        """Name: {{ name }}, Count: {{ count }}, Items: {{ items | join(', ') }}"""

    result = test_prompt("Alice", 3, ["x", "y", "z"])
    assert result == "Name: Alice, Count: 3, Items: x, y, z"
