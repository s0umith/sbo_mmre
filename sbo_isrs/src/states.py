"""State representations for ISRS environment."""

from dataclasses import dataclass
from typing import List, Set, Tuple
import numpy as np
from enum import Enum, auto

from .actions import MultimodalIPPAction

class ISRS_STATE(Enum):
    """Possible states in ISRS environment."""
    UNKNOWN = auto()
    RSGOOD = auto()
    RSBAD = auto()
    RSBEACON = auto()
    RSNEITHER = auto()

@dataclass
class LocationBelief:
    """Belief state for a single location."""
    probs: np.ndarray  # Probabilities for each state

@dataclass
class ISRSWorldState:
    """World state for ISRS environment.
    
    Attributes:
        current: Current location index
        visited: Set of visited location indices
        location_states: List of location states
        cost_expended: Total cost expended so far
    """
    current: int
    visited: Set[int]
    location_states: List[ISRS_STATE]
    cost_expended: float

@dataclass
class ISRSObservation:
    """Observation for ISRS environment.
    
    Attributes:
        current: Current location index
        visited: Set of visited location indices
        location_states: List of observed location states
        cost_expended: Total cost expended so far
        obs_weight: Weight of observation
    """
    current: int
    visited: Set[int]
    location_states: List[ISRS_STATE]
    cost_expended: float
    obs_weight: float = 1.0

    def __hash__(self):
        """Hash function for ISRSObservation."""
        return hash((
            self.current,
            frozenset(self.visited),
            tuple(self.location_states),
            self.cost_expended,
            self.obs_weight
        ))

    def __eq__(self, other):
        """Equality comparison for ISRSObservation."""
        if not isinstance(other, ISRSObservation):
            return False
        return (
            self.current == other.current and
            self.visited == other.visited and
            self.location_states == other.location_states and
            self.cost_expended == other.cost_expended and
            self.obs_weight == other.obs_weight
        )

@dataclass
class ISRSLocationBelief:
    """Belief state for a single location."""
    probs: np.ndarray  # Probability distribution over location states [RSGOOD, RSBAD, RSBEACON, RSNEITHER]

@dataclass
class ISRSBelief:
    """Belief state representation for ISRS environment."""
    current: int  # Current location index
    visited: Set[int]  # Set of visited location indices
    location_beliefs: List[LocationBelief]  # List of location beliefs
    cost_expended: float  # Total cost expended so far

    @classmethod
    def from_world_state(cls, state: ISRSWorldState) -> 'ISRSBelief':
        """Create belief state from world state.
        
        Args:
            state: World state to create belief from
            
        Returns:
            New belief state
        """
        # Initialize uniform beliefs for each location
        location_beliefs = []
        for _ in state.location_states:
            # Uniform distribution over RSGOOD, RSBAD, RSBEACON, RSNEITHER
            probs = np.ones(4) / 4
            location_beliefs.append(LocationBelief(probs))
            
        return cls(
            current=state.current,
            visited=state.visited.copy(),
            location_beliefs=location_beliefs,
            cost_expended=state.cost_expended
        )

    def update(self, a: MultimodalIPPAction, o: ISRSObservation, env) -> 'ISRSBelief':
        """Update belief state with action and observation."""
        if a.visit_location is not None:
            new_location_beliefs = env.belief_update_location_states_visit(
                self.location_beliefs,
                a.visit_location
            )
        else:
            new_location_beliefs = env.belief_update_location_states_sensor(
                self.location_beliefs,
                o.location_states,
                self.current,
                a.sensing_action
            )
        
        return ISRSBelief(
            current=o.current,
            visited=o.visited,
            location_beliefs=new_location_beliefs,
            cost_expended=o.cost_expended
        )

    def sample_state(self, rng: np.random.RandomState) -> ISRSWorldState:
        """Sample a world state from the belief state."""
        location_states = []
        for belief in self.location_beliefs:
            location_states.append(rng.choice(len(belief.probs), p=belief.probs) + 1)  # +1 because enum starts at 1
        
        return ISRSWorldState(
            current=self.current,
            visited=self.visited,
            location_states=location_states,
            cost_expended=self.cost_expended
        )

    def get_metrics(self, true_state: ISRSWorldState) -> Tuple[float, float]:
        """Calculate RMSE and trace metrics.
        
        Args:
            true_state: The true world state to compare against
            
        Returns:
            Tuple of (RMSE, trace) where:
            - RMSE is the root mean square error between belief and true state
            - trace is the trace of the belief covariance matrix
        """
        rmse = 0.0
        trace = 0.0
        
        # Calculate RMSE and individual variances
        variances = []
        for i, (belief, true_state) in enumerate(zip(self.location_beliefs, true_state.location_states)):
            # Calculate RMSE
            true_dist = np.zeros_like(belief.probs)
            true_dist[true_state - 1] = 1.0  # -1 because enum starts at 1
            rmse += np.sum((belief.probs - true_dist) ** 2)
            
            # Calculate variance for this location
            variance = np.sum(belief.probs * (1 - belief.probs))
            variances.append(variance)
        
        # Calculate RMSE
        rmse = np.sqrt(rmse / len(self.location_beliefs))
        
        # Calculate trace of covariance matrix
        # For each location, we have a probability distribution over states
        # The covariance between locations is zero since they're independent
        # So the trace is just the sum of variances
        trace = np.sum(variances)
        
        return rmse, trace 