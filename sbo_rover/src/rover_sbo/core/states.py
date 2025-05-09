"""State representations for the Rover environment."""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict
import numpy as np
from .beliefs import RoverLocationBelief

@dataclass
class RoverPos:
    """Position of the rover in the environment."""
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other) -> bool:
        if not isinstance(other, RoverPos):
            return False
        return self.x == other.x and self.y == other.y

@dataclass
class RoverLocationBelief:
    """Belief about a location's state."""
    probs: np.ndarray  # Probabilities for each possible state
    gp_mean: Optional[float] = None  # GP mean prediction
    gp_var: Optional[float] = None  # GP variance prediction

    def __post_init__(self):
        if not isinstance(self.probs, np.ndarray):
            self.probs = np.array(self.probs)
        if len(self.probs) != 4:  # GOOD_SAMPLE, BAD_SAMPLE, BEACON, EMPTY
            raise ValueError("Location belief must have 4 probabilities")

@dataclass
class RoverBelief:
    """Belief state of the rover."""
    pos: RoverPos
    visited: Set[RoverPos]
    location_beliefs: List[RoverLocationBelief]
    cost_expended: float
    drill_samples: List[Tuple[RoverPos, float]]

    def get_metrics(self, true_state: 'RoverWorldState') -> Tuple[float, float]:
        """Calculate RMSE and trace metrics.
        
        Args:
            true_state: True world state
            
        Returns:
            RMSE and trace metrics
        """
        # Calculate RMSE
        pred_probs = np.array([b.probs for b in self.location_beliefs])
        
        # Convert continuous values to discrete states
        # 0: BAD_SAMPLE (0.0-0.3)
        # 1: BEACON (0.3-0.7)
        # 2: GOOD_SAMPLE (0.7-1.0)
        # 3: EMPTY (default)
        true_states_onehot = np.zeros_like(pred_probs)
        for i, value in enumerate(true_state.location_states):
            if value <= 0.3:
                state = 0  # BAD_SAMPLE
            elif value <= 0.7:
                state = 1  # BEACON
            elif value <= 1.0:
                state = 2  # GOOD_SAMPLE
            else:
                state = 3  # EMPTY
            true_states_onehot[i, state] = 1.0
            
        rmse = np.sqrt(np.mean((pred_probs - true_states_onehot)**2))
        
        # Calculate trace (sum of uncertainties)
        trace = np.sum([np.sum(p * (1 - p)) for p in pred_probs])
        
        return rmse, trace
        
    def sample_state(self, rng: np.random.RandomState) -> 'RoverWorldState':
        """Sample a world state from the belief.
        
        Args:
            rng: Random number generator
            
        Returns:
            Sampled world state
        """
        # Sample location states
        location_states = []
        for belief in self.location_beliefs:
            state = rng.choice(len(belief.probs), p=belief.probs)
            location_states.append(state)
            
        return RoverWorldState(
            pos=self.pos,
            visited=self.visited,
            location_states=np.array(location_states),
            cost_expended=self.cost_expended,
            drill_samples=self.drill_samples
        )

@dataclass
class RoverWorldState:
    """World state of the rover environment."""
    pos: RoverPos
    visited: Set[RoverPos]
    location_states: np.ndarray
    cost_expended: float
    drill_samples: List[Tuple[RoverPos, float]]

    def __post_init__(self):
        if not isinstance(self.location_states, np.ndarray):
            self.location_states = np.array(self.location_states) 

class RoverState:
    """State representation for the Rover environment using continuous values."""
    
    def __init__(
        self,
        position: Tuple[int, int],
        belief: RoverLocationBelief,
        visited: Optional[Dict[Tuple[int, int], bool]] = None
    ):
        """Initialize rover state.
        
        Args:
            position: Current position (x, y)
            belief: Current belief state
            visited: Dictionary of visited positions
        """
        self.position = position
        self.belief = belief
        self.visited = visited or {}
        
    def get_state_vector(self) -> np.ndarray:
        """Get state vector representation.
        
        Returns:
            Array containing:
            - Current position (x, y)
            - Belief map values
            - Uncertainty map values
            - Visited positions mask
        """
        # Get belief and uncertainty maps
        belief_map = self.belief.get_belief_map()
        uncertainty_map = self.belief.get_uncertainty_map()
        
        # Create visited mask
        visited_mask = np.zeros_like(belief_map)
        for pos in self.visited:
            visited_mask[pos[1], pos[0]] = 1.0
            
        # Flatten and concatenate
        state_vector = np.concatenate([
            np.array(self.position),
            belief_map.flatten(),
            uncertainty_map.flatten(),
            visited_mask.flatten()
        ])
        
        return state_vector
        
    def is_terminal(self) -> bool:
        """Check if state is terminal.
        
        Returns:
            True if all positions have been visited
        """
        return len(self.visited) == self.belief.width * self.belief.height
        
    def get_reward(self) -> float:
        """Get reward for current state.
        
        Returns:
            Reward value based on:
            - Information gain (uncertainty reduction)
            - Distance to unexplored areas
            - Penalty for revisiting
        """
        # Information gain component
        uncertainty = self.belief.get_uncertainty_map()
        info_gain = -np.mean(uncertainty)
        
        # Distance to unexplored component
        unvisited_mask = 1.0 - np.array([
            self.visited.get((x, y), 0.0)
            for y in range(self.belief.height)
            for x in range(self.belief.width)
        ]).reshape(self.belief.height, self.belief.width)
        
        # Calculate distances to unvisited cells
        y, x = np.indices(unvisited_mask.shape)
        distances = np.sqrt(
            (x - self.position[0])**2 + (y - self.position[1])**2
        )
        distance_reward = -np.mean(distances * unvisited_mask)
        
        # Penalty for revisiting
        revisit_penalty = -1.0 if self.position in self.visited else 0.0
        
        return info_gain + 0.5 * distance_reward + revisit_penalty 