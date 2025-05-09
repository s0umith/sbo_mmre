"""Action types for ISRS environment."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ISRSSensor(Enum):
    """Sensor types with different efficiencies."""
    LOW = 0.6
    MEDIUM = 0.8
    HIGH = 1.0

@dataclass
class MultimodalIPPAction:
    """Action in ISRS environment.
    
    Attributes:
        visit_location: Location to visit (None if not visiting)
        sensing_action: Sensor action to take (None if not sensing)
        cost: Total cost of the action
    """
    visit_location: Optional[int] = None
    sensing_action: Optional[ISRSSensor] = None
    cost: float = 0.0
    
    def __hash__(self) -> int:
        """Hash function for MultimodalIPPAction."""
        return hash((self.visit_location, self.sensing_action, self.cost))
        
    def __eq__(self, other: object) -> bool:
        """Equality comparison for MultimodalIPPAction."""
        if not isinstance(other, MultimodalIPPAction):
            return False
        return (self.visit_location == other.visit_location and
                self.sensing_action == other.sensing_action and
                self.cost == other.cost) 