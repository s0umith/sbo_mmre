"""Belief state for ISRS environment."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# Local imports
from .states import ISRSWorldState, ISRSObservation, ISRS_STATE
from .belief_types import ISRSLocationBelief
from .belief_updates import calculate_belief_metrics
from .actions import MultimodalIPPAction

@dataclass
class ISRSBelief:
    """Belief state for ISRS environment.
    
    Attributes:
        current: Current location
        visited: Set of visited locations
        location_beliefs: List of location beliefs
        cost_expended: Total cost expended
    """
    current: int
    visited: set
    location_beliefs: List[ISRSLocationBelief]
    cost_expended: float
    
    def sample_state(self, rng: np.random.RandomState) -> ISRSWorldState:
        """Sample world state from belief.
        
        Args:
            rng: Random number generator
            
        Returns:
            Sampled world state
        """
        # Sample location states
        location_states = []
        for belief in self.location_beliefs:
            location_states.append(belief.sample_state(rng))
            
        return ISRSWorldState(
            current=self.current,
            visited=self.visited,
            location_states=location_states,
            cost_expended=self.cost_expended
        )
        
    def get_metrics(self, true_state: ISRSWorldState) -> tuple:
        """Calculate belief metrics.
        
        Args:
            true_state: True world state
            
        Returns:
            Tuple of (RMSE, entropy)
        """
        return calculate_belief_metrics(self, true_state)
        
    def update(
        self,
        action: MultimodalIPPAction,
        observation: ISRSObservation,
        state: Optional[ISRSWorldState] = None
    ) -> 'ISRSBelief':
        """Update belief based on action and observation.
        
        Args:
            action: Action taken
            observation: Observation received
            state: True state (optional, for debugging)
            
        Returns:
            Updated belief
        """
        # Create copy of current belief
        new_belief = ISRSBelief(
            current=observation.current,
            visited=observation.visited,
            location_beliefs=self.location_beliefs.copy(),
            cost_expended=observation.cost_expended
        )
        
        # Update belief based on action type
        if action.visit_location is not None:
            # Perfect observation at target location
            new_belief.location_beliefs[action.visit_location] = ISRSLocationBelief.from_observation(
                observation.location_states[action.visit_location]
            )
        elif action.sensing_action is not None:
            # Update belief based on sensor efficiency
            if observation.location_states[observation.current] != ISRS_STATE.UNKNOWN:
                # Successful sensing
                new_belief.location_beliefs[observation.current] = ISRSLocationBelief.from_observation(
                    observation.location_states[observation.current]
                )
            else:
                # Failed sensing
                new_belief.location_beliefs[observation.current].update_with_failure(action.sensing_action.value)
        
        return new_belief 