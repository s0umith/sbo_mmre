"""Enhanced reward components for POMCP policies."""

from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .states import ISRSWorldState, ISRSBelief
from .actions import MultimodalIPPAction
from .env import ISRS_STATE

@dataclass
class RewardComponents:
    """Container for reward component values.
    
    Attributes:
        base_reward: Base reward from POMDP
        info_gain: Information gain component
        exploration_bonus: Exploration bonus
        progress_factor: Progressive reward scaling
        distance_penalty: Distance-based penalty
        belief_bonus: Belief state bonus
    """
    base_reward: float = 0.0
    info_gain: float = 0.0
    exploration_bonus: float = 0.0
    progress_factor: float = 0.0
    distance_penalty: float = 0.0
    belief_bonus: float = 0.0

class EnhancedRewardCalculator:
    """Calculates enhanced rewards incorporating multiple components.
    
    Attributes:
        info_gain_weight: Weight for information gain
        exploration_weight: Weight for exploration bonus
        progressive_weight: Weight for progressive rewards
        distance_penalty_weight: Weight for distance penalty
        belief_weight: Weight for belief bonus
        exploration_bonus: Base exploration bonus
        progressive_factor: Base progressive factor
    """
    
    def __init__(
        self,
        info_gain_weight: float = 0.4,
        exploration_weight: float = 0.6,
        progressive_weight: float = 0.4,
        distance_penalty_weight: float = 0.1,
        belief_weight: float = 0.3,
        exploration_bonus: float = 2.0,
        progressive_factor: float = 1.2
    ) -> None:
        """Initialize reward calculator.
        
        Args:
            info_gain_weight: Weight for information gain
            exploration_weight: Weight for exploration bonus
            progressive_weight: Weight for progressive rewards
            distance_penalty_weight: Weight for distance penalty
            belief_weight: Weight for belief bonus
            exploration_bonus: Base exploration bonus
            progressive_factor: Base progressive factor
        """
        self.info_gain_weight = info_gain_weight
        self.exploration_weight = exploration_weight
        self.progressive_weight = progressive_weight
        self.distance_penalty_weight = distance_penalty_weight
        self.belief_weight = belief_weight
        self.exploration_bonus = exploration_bonus
        self.progressive_factor = progressive_factor

    def calculate_reward(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction,
        next_state: ISRSWorldState,
        belief: Optional[ISRSBelief] = None,
        base_reward: float = 0.0
    ) -> Tuple[float, RewardComponents]:
        """Calculate enhanced reward incorporating multiple components.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            belief: Current belief state
            base_reward: Base reward from POMDP
            
        Returns:
            Tuple of (total reward, reward components)
        """
        # Calculate individual components
        components = RewardComponents(base_reward=base_reward)
        
        # Information gain
        components.info_gain = self._calculate_info_gain(state, action, next_state)
        
        # Exploration bonus
        components.exploration_bonus = self._calculate_exploration_bonus(state, action)
        
        # Progressive reward scaling
        components.progress_factor = self._calculate_progress_factor(state)
        
        # Distance penalty
        components.distance_penalty = self._calculate_distance_penalty(state, action)
        
        # Belief bonus
        if belief is not None:
            components.belief_bonus = self._calculate_belief_bonus(state, action, belief)
        
        # Get adaptive weights
        weights = self._get_adaptive_weights(state)
        
        # Calculate total reward
        total_reward = (
            components.base_reward +
            weights['info_gain'] * components.info_gain +
            weights['exploration'] * components.exploration_bonus +
            weights['progressive'] * components.progress_factor -
            weights['distance'] * components.distance_penalty +
            weights['belief'] * components.belief_bonus
        )
        
        return total_reward, components

    def _calculate_info_gain(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction,
        next_state: ISRSWorldState
    ) -> float:
        """Calculate information gain from action.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Information gain value
        """
        if action.sensing_action is None:
            return 0.0
            
        # Calculate entropy reduction
        current_entropy = self._calculate_state_entropy(state)
        next_entropy = self._calculate_state_entropy(next_state)
        return max(0, current_entropy - next_entropy)

    def _calculate_exploration_bonus(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate exploration bonus for visiting new locations.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Exploration bonus value
        """
        if action.visit_location is None or action.visit_location in state.visited:
            return 0.0
            
        # Calculate bonus based on distance to nearest unvisited location
        min_dist = float('inf')
        for loc in range(len(state.location_states)):
            if loc not in state.visited:
                dist = np.linalg.norm(
                    np.array(state.location_metadata[state.current]) - 
                    np.array(state.location_metadata[loc])
                )
                min_dist = min(min_dist, dist)
                
        return self.exploration_bonus / (1 + min_dist)

    def _calculate_progress_factor(self, state: ISRSWorldState) -> float:
        """Calculate progressive reward scaling factor.
        
        Args:
            state: Current state
            
        Returns:
            Progress factor
        """
        return self.progressive_factor ** (len(state.visited) / len(state.location_states))

    def _calculate_distance_penalty(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate distance-based penalty.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Distance penalty
        """
        if action.visit_location is not None:
            distance = np.linalg.norm(
                np.array(state.location_metadata[state.current]) - 
                np.array(state.location_metadata[action.visit_location])
            )
            return np.sqrt(distance)
        return 0.0

    def _calculate_belief_bonus(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction,
        belief: ISRSBelief
    ) -> float:
        """Calculate bonus based on belief state.
        
        Args:
            state: Current state
            action: Action taken
            belief: Current belief state
            
        Returns:
            Belief bonus value
        """
        # Calculate belief state similarity
        belief_similarity = np.mean(
            belief.location_states == state.location_states
        )
        
        # Calculate uncertainty reduction
        uncertainty_reduction = 0.0
        if action.sensing_action is not None:
            current_uncertainty = np.mean(
                belief.location_states == ISRS_STATE.RSUNKNOWN
            )
            next_uncertainty = np.mean(
                next_state.location_states == ISRS_STATE.RSUNKNOWN
            )
            uncertainty_reduction = max(0, current_uncertainty - next_uncertainty)
        
        return belief_similarity + uncertainty_reduction

    def _calculate_state_entropy(self, state: ISRSWorldState) -> float:
        """Calculate entropy of state's location states.
        
        Args:
            state: Current state
            
        Returns:
            State entropy
        """
        # Calculate entropy of rock states
        probs = np.array([
            np.mean(state.location_states == ISRS_STATE.RSGOOD),
            np.mean(state.location_states == ISRS_STATE.RSBAD),
            np.mean(state.location_states == ISRS_STATE.RSUNKNOWN)
        ])
        probs = probs[probs > 0]  # Remove zero probabilities
        return -np.sum(probs * np.log2(probs))

    def _get_adaptive_weights(self, state: ISRSWorldState) -> Dict[str, float]:
        """Get adaptive weights based on state.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary of weights
        """
        # Calculate progress
        progress = len(state.visited) / len(state.location_states)
        
        # Adjust weights based on progress
        return {
            'info_gain': self.info_gain_weight * (1 - progress),
            'exploration': self.exploration_weight * (1 - progress),
            'progressive': self.progressive_weight * progress,
            'distance': self.distance_penalty_weight,
            'belief': self.belief_weight
        } 