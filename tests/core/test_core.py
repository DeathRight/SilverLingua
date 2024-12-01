"""Tests for core components, run in hierarchical order."""

import pytest


@pytest.mark.core
@pytest.mark.atoms
@pytest.mark.dependency()
def test_atoms():
    """Run all atom tests."""
    pytest.main(["-v", "tests/core/atoms"])


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.dependency(depends=["test_atoms"])
def test_molecules():
    """Run all molecule tests."""
    pytest.main(["-v", "tests/core/molecules"])


@pytest.mark.core
@pytest.mark.organisms
@pytest.mark.dependency(depends=["test_molecules"])
def test_organisms():
    """Run all organism tests."""
    pytest.main(["-v", "tests/core/organisms"])


@pytest.mark.core
@pytest.mark.templates
@pytest.mark.dependency(depends=["test_organisms"])
def test_templates():
    """Run all template tests."""
    pytest.main(["-v", "tests/core/templates"])
