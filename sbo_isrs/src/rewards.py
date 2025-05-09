"""Reward representation for ISRS environment."""

from dataclasses import dataclass
from typing import List
from .state import Position

@dataclass
class ISRSReward:
    """Reward in the ISRS environment.
    
    Attributes:
        value: Reward value
        pos: Position where reward was obtained
    """
    value: float
    pos: Position
    
    def __hash__(self):
        return hash((self.value, self.pos))
        
    def __eq__(self, other):
        return isinstance(other, ISRSReward) and \
               self.value == other.value and \
               self.pos == other.pos

    @classmethod
    def get_all_rewards(cls) -> List['ISRSReward']:
        """Get all possible rewards."""
        return [
            cls(value=0.0, pos=Position(0, 0)),
            cls(value=1.0, pos=Position(0, 0)),
            cls(value=0.0, pos=Position(0, 0)),
            cls(value=1.0, pos=Position(0, 0))
        ] 