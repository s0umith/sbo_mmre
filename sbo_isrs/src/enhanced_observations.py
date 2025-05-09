"""Enhanced observation handling with GP integration and similarity metrics."""

from typing import List, Optional, Set
import numpy as np
from dataclasses import dataclass

from .env import ISRS_STATE

@dataclass
class EnhancedObservation:
    """Enhanced observation class with similarity metrics and GP integration.
    
    Attributes:
        current: Current location
        visited: Set of visited locations
        location_states: Array of location states
        cost_expended: Total cost expended
        obs_weight: Weight of this observation
        gp_features: GP feature vector
        info_gain: Information gain from this observation
        uncertainty: Uncertainty in observation
    """
    current: int
    visited: Set[int]
    location_states: np.ndarray
    cost_expended: float
    obs_weight: float
    gp_features: Optional[np.ndarray] = None
    info_gain: float = 0.0
    uncertainty: float = 0.0

    def similarity(self, other: 'EnhancedObservation') -> float:
        """Calculate similarity between observations.
        
        Args:
            other: Other observation to compare with
            
        Returns:
            Similarity score between 0 and 1
        """
        # Location state similarity
        loc_sim = np.mean(self.location_states == other.location_states)
        
        # Visited locations similarity
        visited_sim = len(self.visited & other.visited) / max(len(self.visited), len(other.visited))
        
        # GP feature similarity if available
        gp_sim = 0.0
        if self.gp_features is not None and other.gp_features is not None:
            gp_sim = np.exp(-np.linalg.norm(self.gp_features - other.gp_features))
            
        # Information gain similarity
        info_sim = 1.0 - abs(self.info_gain - other.info_gain)
        
        # Uncertainty similarity
        uncert_sim = 1.0 - abs(self.uncertainty - other.uncertainty)
            
        return (
            0.3 * loc_sim +
            0.2 * visited_sim +
            0.2 * gp_sim +
            0.2 * info_sim +
            0.1 * uncert_sim
        )

    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for GP integration.
        
        Returns:
            Feature vector
        """
        features = []
        
        # Location state features
        features.extend([
            np.mean(self.location_states == ISRS_STATE.RSGOOD),
            np.mean(self.location_states == ISRS_STATE.RSBAD),
            np.mean(self.location_states == ISRS_STATE.RSUNKNOWN)
        ])
        
        # Visited locations features
        features.append(len(self.visited) / len(self.location_states))
        
        # Cost features
        features.append(self.cost_expended)
        
        # Information gain features
        features.append(self.info_gain)
        features.append(self.uncertainty)
        
        return np.array(features)

class ObservationWidener:
    """Class for handling observation widening with enhanced features.
    
    Attributes:
        k: Widening parameter
        alpha: Widening exponent
        threshold: Similarity threshold
        max_observations: Maximum number of observations
        observations: List of stored observations
        gp_model: Optional GP model for observation prediction
    """
    
    def __init__(
        self,
        k: float = 0.5,
        alpha: float = 0.5,
        threshold: float = 0.1,
        max_observations: int = 1000,
        gp_noise_level: float = 1e-4,
        gp_n_restarts: int = 10
    ) -> None:
        """Initialize observation widener.
        
        Args:
            k: Widening parameter
            alpha: Widening exponent
            threshold: Similarity threshold
            max_observations: Maximum number of observations
            gp_noise_level: Noise level for GP
            gp_n_restarts: Number of restarts for GP optimization
        """
        self.k = k
        self.alpha = alpha
        self.threshold = threshold
        self.max_observations = max_observations
        self.observations = []
        self.gp_noise_level = gp_noise_level
        self.gp_n_restarts = gp_n_restarts
        self.gp_model = None

    def add_observation(
        self,
        obs: EnhancedObservation,
        info_gain: float = 0.0,
        uncertainty: float = 0.0
    ) -> EnhancedObservation:
        """Add new observation with widening.
        
        Args:
            obs: New observation to add
            info_gain: Information gain from observation
            uncertainty: Uncertainty in observation
            
        Returns:
            Processed observation
        """
        # Update observation with additional information
        obs.info_gain = info_gain
        obs.uncertainty = uncertainty
        
        # Calculate GP features
        obs.gp_features = obs.get_feature_vector()
        
        if len(self.observations) >= self.max_observations:
            return self._merge_with_most_similar(obs)
            
        # Find most similar observation
        most_similar = self._find_most_similar(obs)
        
        if most_similar is None or obs.similarity(most_similar) < self.threshold:
            self.observations.append(obs)
            return obs
        else:
            return self._merge_observations(obs, most_similar)

    def _find_most_similar(self, obs: EnhancedObservation) -> Optional[EnhancedObservation]:
        """Find most similar observation to given observation.
        
        Args:
            obs: Observation to compare against
            
        Returns:
            Most similar observation or None if no observations
        """
        if not self.observations:
            return None
            
        similarities = [obs.similarity(other) for other in self.observations]
        return self.observations[np.argmax(similarities)]

    def _merge_observations(
        self,
        obs1: EnhancedObservation,
        obs2: EnhancedObservation
    ) -> EnhancedObservation:
        """Merge two similar observations.
        
        Args:
            obs1: First observation
            obs2: Second observation
            
        Returns:
            Merged observation
        """
        # Calculate weights
        w1 = obs1.obs_weight
        w2 = obs2.obs_weight
        total_weight = w1 + w2
        
        # Merge location states
        merged_states = np.where(
            obs1.location_states == obs2.location_states,
            obs1.location_states,
            ISRS_STATE.RSUNKNOWN
        )
        
        # Create merged observation
        merged = EnhancedObservation(
            current=obs1.current,
            visited=obs1.visited | obs2.visited,
            location_states=merged_states,
            cost_expended=(w1 * obs1.cost_expended + w2 * obs2.cost_expended) / total_weight,
            obs_weight=total_weight,
            gp_features=(w1 * obs1.gp_features + w2 * obs2.gp_features) / total_weight,
            info_gain=(w1 * obs1.info_gain + w2 * obs2.info_gain) / total_weight,
            uncertainty=(w1 * obs1.uncertainty + w2 * obs2.uncertainty) / total_weight
        )
        
        # Replace old observation with merged one
        self.observations.remove(obs2)
        self.observations.append(merged)
        
        return merged

    def _merge_with_most_similar(self, obs: EnhancedObservation) -> EnhancedObservation:
        """Merge observation with most similar existing observation.
        
        Args:
            obs: Observation to merge
            
        Returns:
            Merged observation
        """
        most_similar = self._find_most_similar(obs)
        if most_similar is None:
            return obs
            
        return self._merge_observations(obs, most_similar)

    def get_widened_observations(self) -> List[EnhancedObservation]:
        """Get list of widened observations.
        
        Returns:
            List of widened observations
        """
        return self.observations

    def clear(self) -> None:
        """Clear all stored observations."""
        self.observations = [] 