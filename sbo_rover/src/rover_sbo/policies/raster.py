"""Raster policy for the Rover environment."""

from typing import Tuple
import numpy as np

class RasterPolicy:
    """Raster policy that follows a systematic pattern of exploration.
    
    This policy implements a raster scan pattern similar to the Julia implementation,
    moving in a systematic way across the grid while periodically drilling.
    """
    
    def __init__(self, grid_size: Tuple[int, int], drill_interval: int = 30):
        """Initialize the raster policy.
        
        Args:
            grid_size: Size of the grid (width, height)
            drill_interval: Number of steps between drill actions
        """
        self.grid_size = grid_size
        self.drill_interval = drill_interval
        self.step_count = 0
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from raster policy.
        
        Args:
            state: Current state vector
            
        Returns:
            Action vector (dx, dy)
        """
        self.step_count += 1
        x, y = state[:2]  # Extract position from state
        
        # Check if it's time to drill
        if self.step_count % self.drill_interval == 0:
            if x == 0 and y == 0:
                return np.array([1, 0])  # Move right
            return np.array([0, 0])  # Drill
            
        # In an odd column
        if x % 2 == 0:
            # At the top of the column
            if y == self.grid_size[1] - 1:
                return np.array([1, 0])  # Move right
            else:
                return np.array([0, 1])  # Move up
        # In an even column
        else:
            # At the bottom of the column
            if y == 0:
                if x == 0:
                    return np.array([1, 0])  # Move right
                else:
                    return np.array([1, 0])  # Move right
            else:
                return np.array([0, -1])  # Move down
                
    def update(self, *args, **kwargs) -> None:
        """Update policy parameters (not used for raster policy)."""
        pass 