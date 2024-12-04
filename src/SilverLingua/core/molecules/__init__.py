"""
Molecules represent self-contained conceptual units that combine atoms to create more complex 
but focused components. They focus on "what something is" rather than "what it does."

See [Design Principles - Molecules](/design-principles/#molecules) for more details.
"""

from .link import Link
from .notion import Notion

__all__ = ["Notion", "Link"]
