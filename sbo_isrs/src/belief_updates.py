"""Belief update functions for ISRS environment."""

from typing import TYPE_CHECKING, Optional
import numpy as np

# Local imports
from .states import ISRS_STATE, ISRSWorldState

if TYPE_CHECKING:
    from .belief import ISRSBelief, ISRSLocationBelief
    from .belief_types import ISRSLocationBelief

def update_belief_visit(
    belief: 'ISRSLocationBelief',
    location: int,
    observation: Optional[ISRS_STATE] = None,
) -> 'ISRSLocationBelief':
    """Update belief after visiting a location.
    
    Args:
        belief: Current belief state
        location: Location being visited
        observation: Optional observation at the location
        
    Returns:
        Updated belief state
    """
    new_belief = belief.copy()
    new_belief.update_visit(observation)
    return new_belief

def update_belief_sensor(
    belief: 'ISRSLocationBelief',
    location: int,
    observation: ISRS_STATE,
    sensor_efficiency: float,
) -> 'ISRSLocationBelief':
    """Update belief after sensing a location.
    
    Args:
        belief: Current belief state
        location: Location being sensed
        observation: Observation from sensor
        sensor_efficiency: Efficiency of the sensor
        
    Returns:
        Updated belief state
    """
    new_belief = belief.copy()
    new_belief.update_sensor(observation, sensor_efficiency)
    return new_belief

def calculate_belief_metrics(
    belief: 'ISRSBelief',
    true_state: ISRSWorldState,
) -> tuple[float, float]:
    """Calculate belief metrics for a given true state.
    
    Args:
        belief: The current belief state
        true_state: The true state of the environment
        
    Returns:
        tuple[float, float]: RMSE and entropy of the belief distribution
    """
    # Define mapping from ISRS_STATE to belief state indices
    # UNKNOWN is not included in belief state
    # Order in belief state is [RSGOOD, RSBAD, RSBEACON, RSNEITHER]
    state_to_idx = {
        ISRS_STATE.RSGOOD: 0,
        ISRS_STATE.RSBAD: 1,
        ISRS_STATE.RSBEACON: 2,
        ISRS_STATE.RSNEITHER: 3
    }
    
    # Calculate RMSE between belief and true state
    rmse = 0.0
    for i, (belief_state, true_value) in enumerate(zip(belief.location_beliefs, true_state.location_states)):
        # Skip unknown states
        if true_value == ISRS_STATE.UNKNOWN:
            continue
            
        # Convert true state to one-hot encoding
        true_probs = np.zeros_like(belief_state.probs)
        true_probs[state_to_idx[true_value]] = 1.0
        
        # Calculate squared error for this location
        rmse += np.sum((belief_state.probs - true_probs) ** 2)
    
    # Normalize RMSE by number of non-unknown states
    num_known_states = sum(1 for state in true_state.location_states if state != ISRS_STATE.UNKNOWN)
    if num_known_states > 0:
        rmse = np.sqrt(rmse / num_known_states)
    else:
        rmse = 0.0
    
    # Calculate entropy of belief distribution
    entropy = 0.0
    for belief_state in belief.location_beliefs:
        # Only consider non-zero probabilities to avoid log(0)
        non_zero_probs = belief_state.probs[belief_state.probs > 0]
        if len(non_zero_probs) > 0:
            entropy -= np.sum(non_zero_probs * np.log2(non_zero_probs))
    
    return rmse, entropy 