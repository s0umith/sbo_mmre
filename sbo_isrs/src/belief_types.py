"""Belief type definitions for ISRS environment."""

import numpy as np
from dataclasses import dataclass

# Local imports
from .states import ISRS_STATE

@dataclass
class ISRSLocationBelief:
    """Belief about a single location in ISRS environment.
    
    Attributes:
        probs: Probability distribution over states
    """
    probs: np.ndarray = None
    
    def __post_init__(self):
        """Initialize uniform belief if none provided."""
        if self.probs is None:
            self.probs = np.ones(5) / 5  # Uniform over all states
            
    def sample_state(self, rng: np.random.RandomState) -> ISRS_STATE:
        """Sample state from belief distribution.
        
        Args:
            rng: Random number generator
            
        Returns:
            Sampled state
        """
        return ISRS_STATE(rng.choice(len(self.probs), p=self.probs) + 1)  # +1 because enum starts at 1
        
    def update_with_failure(self, sensor_efficiency: float) -> None:
        """Update belief after failed sensing.
        
        Args:
            sensor_efficiency: Sensor efficiency
        """
        # Decrease probability of UNKNOWN state
        self.probs[ISRS_STATE.UNKNOWN.value - 1] *= (1 - sensor_efficiency)  # -1 because enum starts at 1
        
        # Normalize probabilities
        self.probs /= self.probs.sum()
        
    @classmethod
    def from_observation(cls, state: ISRS_STATE) -> 'ISRSLocationBelief':
        """Create belief from observed state.
        
        Args:
            state: Observed state
            
        Returns:
            Location belief
        """
        probs = np.zeros(5)  # 5 states: UNKNOWN, RSGOOD, RSBAD, RSBEACON, RSNEITHER
        probs[state.value - 1] = 1.0  # -1 because enum starts at 1
        return cls(probs=probs)
        
    def update_visit(self, true_state: ISRS_STATE) -> None:
        """Update belief after visiting location.
        
        Args:
            true_state: True state of location
        """
        # Reset probabilities
        self.probs = np.zeros_like(self.probs)
        self.probs[true_state.value - 1] = 1.0  # -1 because enum starts at 1
        
    def update_sensor(self, observation: ISRS_STATE, efficiency: float) -> None:
        """Update belief after sensor observation.
        
        Args:
            observation: Observed state
            efficiency: Sensor efficiency
        """
        # Calculate likelihood of observation given each state
        likelihoods = {
            ISRS_STATE.RSGOOD: efficiency if observation == ISRS_STATE.RSGOOD else (1 - efficiency) / 3,
            ISRS_STATE.RSBAD: efficiency if observation == ISRS_STATE.RSBAD else (1 - efficiency) / 3,
            ISRS_STATE.RSBEACON: efficiency if observation == ISRS_STATE.RSBEACON else (1 - efficiency) / 3,
            ISRS_STATE.RSNEITHER: efficiency if observation == ISRS_STATE.RSNEITHER else (1 - efficiency) / 3
        }
        
        # Update probabilities using Bayes rule
        total = (
            self.probs[ISRS_STATE.RSGOOD.value] * likelihoods[ISRS_STATE.RSGOOD] +
            self.probs[ISRS_STATE.RSBAD.value] * likelihoods[ISRS_STATE.RSBAD] +
            self.probs[ISRS_STATE.RSBEACON.value] * likelihoods[ISRS_STATE.RSBEACON] +
            self.probs[ISRS_STATE.RSNEITHER.value] * likelihoods[ISRS_STATE.RSNEITHER]
        )
        
        self.probs[ISRS_STATE.RSGOOD.value] = self.probs[ISRS_STATE.RSGOOD.value] * likelihoods[ISRS_STATE.RSGOOD] / total
        self.probs[ISRS_STATE.RSBAD.value] = self.probs[ISRS_STATE.RSBAD.value] * likelihoods[ISRS_STATE.RSBAD] / total
        self.probs[ISRS_STATE.RSBEACON.value] = self.probs[ISRS_STATE.RSBEACON.value] * likelihoods[ISRS_STATE.RSBEACON] / total
        self.probs[ISRS_STATE.RSNEITHER.value] = self.probs[ISRS_STATE.RSNEITHER.value] * likelihoods[ISRS_STATE.RSNEITHER] / total 