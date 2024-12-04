"""
Templates define core interfaces and abstract base classes that ensure consistency across 
the system. They establish patterns that other components must follow while keeping 
implementation details flexible.

See [Design Principles - Templates](/design-principles/#templates) for more details.
"""

from .agent import Agent
from .model import Model

__all__ = ["Agent", "Model"]
