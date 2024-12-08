import pytest

from silverlingua.core.atoms import Memory
from silverlingua.core.molecules import Link, Notion


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_link_initialization():
    """Test basic link initialization."""
    # Test with Notion
    notion = Notion(content="Hello", role="SYSTEM")
    link = Link(content=notion)
    assert link.content == notion
    assert link.parent is None
    assert len(link.children) == 0

    # Test with Memory
    memory = Memory(content="Hello")
    link = Link(content=memory)
    assert link.content == memory
    assert link.parent is None
    assert len(link.children) == 0


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_link_hierarchy():
    """Test link hierarchy operations."""
    root = Link(content=Notion(content="Root", role="SYSTEM"))
    child1 = Link(content=Notion(content="Child 1", role="SYSTEM"))
    child2 = Link(content=Notion(content="Child 2", role="SYSTEM"))
    grandchild = Link(content=Notion(content="Grandchild", role="SYSTEM"))

    # Add children
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(grandchild)

    # Test parent-child relationships
    assert child1.parent == root
    assert child2.parent == root
    assert grandchild.parent == child1
    assert len(root.children) == 2
    assert len(child1.children) == 1
    assert len(child2.children) == 0

    # Remove child
    root.remove_child(child1)
    assert child1.parent is None
    assert len(root.children) == 1


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_link_path():
    """Test path-related properties."""
    root = Link(content=Notion(content="Root", role="SYSTEM"))
    child = Link(content=Notion(content="Child", role="SYSTEM"))
    grandchild = Link(content=Notion(content="Grandchild", role="SYSTEM"))

    root.add_child(child)
    child.add_child(grandchild)

    # Test path
    path = grandchild.path
    assert len(path) == 3
    assert path[0] == grandchild
    assert path[1] == child
    assert path[2] == root

    # Test root
    assert grandchild.root == root
    assert child.root == root
    assert root.root == root

    # Test depth
    assert root.depth == 1
    assert child.depth == 2
    assert grandchild.depth == 3


@pytest.mark.core
@pytest.mark.molecules
@pytest.mark.unit
def test_link_properties():
    """Test various link properties."""
    root = Link(content=Notion(content="Root", role="SYSTEM"))
    child = Link(content=Notion(content="Child", role="SYSTEM"))
    grandchild = Link(content=Notion(content="Grandchild", role="SYSTEM"))

    root.add_child(child)
    child.add_child(grandchild)

    # Test is_root
    assert root.is_root
    assert not child.is_root
    assert not grandchild.is_root

    # Test is_leaf
    assert grandchild.is_leaf
    assert not child.is_leaf
    assert not root.is_leaf

    # Test is_branch
    assert not grandchild.is_branch
    assert child.is_branch
    assert root.is_branch

    # Test path_string
    assert root.path_string == "SYSTEM: Root"
    assert child.path_string == "SYSTEM: Root>SYSTEM: Child"
    assert grandchild.path_string == "SYSTEM: Root>SYSTEM: Child>SYSTEM: Grandchild"
