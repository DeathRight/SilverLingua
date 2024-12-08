from importlib.util import find_spec

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "openai: mark test as requiring OpenAI")
    config.addinivalue_line("markers", "anthropic: mark test as requiring Anthropic")


def pytest_runtest_setup(item):
    for marker in item.iter_markers():
        if marker.name == "openai" and find_spec("openai") is None:
            pytest.skip("OpenAI package not installed")
        elif marker.name == "anthropic" and find_spec("anthropic") is None:
            pytest.skip("Anthropic package not installed")
