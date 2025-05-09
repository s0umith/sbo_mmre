"""Action definitions for the Rover environment."""

from enum import Enum, IntEnum, auto
from typing import Tuple
from dataclasses import dataclass

class RoverActionType(Enum):
    """Types of actions the rover can take."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    WAIT = auto()
    NE = auto()  # Northeast
    NW = auto()  # Northwest
    SE = auto()  # Southeast
    SW = auto()  # Southwest
    DRILL = auto()

class RoverAction:
    """Action that can be taken by the rover."""
    
    def __init__(self, action_type: RoverActionType, target_pos: Tuple[int, int] = None):
        """Initialize action.
        
        Args:
            action_type: Type of action
            target_pos: Target position for drill action
        """
        self.action_type = action_type
        self.target_pos = target_pos
        
    def __str__(self) -> str:
        """String representation of action."""
        if self.action_type == RoverActionType.DRILL:
            return f"Drill at {self.target_pos}"
        return self.action_type.name

# Direction vectors for each action
DIRECTIONS = {
    RoverActionType.UP: (0, 1),
    RoverActionType.DOWN: (0, -1),
    RoverActionType.LEFT: (-1, 0),
    RoverActionType.RIGHT: (1, 0),
    RoverActionType.WAIT: (0, 0),
    RoverActionType.NE: (1, 1),
    RoverActionType.NW: (-1, 1),
    RoverActionType.SE: (1, -1),
    RoverActionType.SW: (-1, -1),
    RoverActionType.DRILL: (0, 0)
}

# Action indices for mapping
ACTION_INDICES = {
    RoverActionType.UP: 1,
    RoverActionType.DOWN: 2,
    RoverActionType.LEFT: 3,
    RoverActionType.RIGHT: 4,
    RoverActionType.WAIT: 5,
    RoverActionType.NE: 6,
    RoverActionType.NW: 7,
    RoverActionType.SE: 8,
    RoverActionType.SW: 9,
    RoverActionType.DRILL: 10
}

class RoverSensorType(IntEnum):
    """Types of sensors available to the Rover."""
    CAMERA = 1
    SPECTROMETER = 2
    DRILL = 3

@dataclass
class RoverSensor:
    """Sensor configuration for the Rover environment."""
    efficiency: float  # Sensor efficiency parameter
    cost: float  # Cost of using the sensor
    range: float  # Maximum sensing range 