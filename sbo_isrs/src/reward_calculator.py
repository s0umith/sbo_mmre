"""Reward calculation for ISRS environment."""

import numpy as np

from .actions import MultimodalIPPAction
from .states import ISRSWorldState, ISRSBelief, ISRS_STATE

class RewardCalculator:
    """Calculates rewards for ISRS environment.
    
    Attributes:
        info_gain_weight: Weight for information gain reward
        distance_penalty_weight: Weight for distance penalty
        exploration_bonus: Bonus for exploring new locations
        progressive_factor: Factor for progressive rewards
        belief_weight: Weight for belief-based rewards
        uncertainty_threshold: Threshold for uncertainty-based rewards
    """
    
    def __init__(
        self,
        info_gain_weight: float = 0.3,
        distance_penalty_weight: float = 0.2,
        exploration_bonus: float = 5.0,
        progressive_factor: float = 1.2,
        belief_weight: float = 0.2,
        uncertainty_threshold: float = 0.1
    ) -> None:
        """Initialize reward calculator.
        
        Args:
            info_gain_weight: Weight for information gain reward
            distance_penalty_weight: Weight for distance penalty
            exploration_bonus: Bonus for exploring new locations
            progressive_factor: Factor for progressive rewards
            belief_weight: Weight for belief-based rewards
            uncertainty_threshold: Threshold for uncertainty-based rewards
        """
        self.info_gain_weight = info_gain_weight
        self.distance_penalty_weight = distance_penalty_weight
        self.exploration_bonus = exploration_bonus
        self.progressive_factor = progressive_factor
        self.belief_weight = belief_weight
        self.uncertainty_threshold = uncertainty_threshold

    def calculate_reward(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction,
        belief: ISRSBelief,
        next_state: ISRSWorldState,
        next_belief: ISRSBelief
    ) -> float:
        """Calculate total reward for action.
        
        Args:
            state: Current state
            action: Action taken
            belief: Current belief
            next_state: Next state
            next_belief: Next belief
            
        Returns:
            Total reward value
        """
        # Base reward for visiting good/bad rock samples
        base_reward = self._calculate_base_reward(state, action)
        
        # Information gain reward
        info_gain = self._calculate_info_gain(belief, next_belief)
        
        # Distance penalty
        distance_penalty = self._calculate_distance_penalty(state, action)
        
        # Exploration bonus
        exploration = self._calculate_exploration_bonus(state, action)
        
        # Progressive reward
        progressive = self._calculate_progressive_reward(state, action)
        
        # Belief-based reward
        belief_reward = self._calculate_belief_reward(belief, next_belief)
        
        # Combine all rewards
        total_reward = (
            base_reward +
            self.info_gain_weight * info_gain -
            self.distance_penalty_weight * distance_penalty +
            exploration +
            progressive +
            self.belief_weight * belief_reward
        )
        
        return total_reward

    def _calculate_base_reward(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate base reward for visiting rock samples.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Base reward value
        """
        if action.visit_location is not None:
            if state.location_states[action.visit_location] == ISRS_STATE.RSGOOD:
                return 1.0
            elif state.location_states[action.visit_location] == ISRS_STATE.RSBAD:
                return -1.0
        return 0.0

    def _calculate_info_gain(
        self,
        belief: ISRSBelief,
        next_belief: ISRSBelief
    ) -> float:
        """Calculate information gain reward.
        
        Args:
            belief: Current belief
            next_belief: Next belief
            
        Returns:
            Information gain reward
        """
        # Calculate uncertainty reduction
        current_rmse, current_trace = belief.get_metrics(belief.sample_state(np.random.RandomState()))
        next_rmse, next_trace = next_belief.get_metrics(next_belief.sample_state(np.random.RandomState()))
        
        # Information gain is the reduction in uncertainty
        info_gain = (current_trace - next_trace) / max(1.0, current_trace)
        return max(0.0, info_gain)

    def _calculate_distance_penalty(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate distance penalty.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Distance penalty
        """
        if action.visit_location is not None:
            # Calculate distance between current and target location
            distance = abs(state.current - action.visit_location)
            return distance
        return 0.0

    def _calculate_exploration_bonus(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate exploration bonus.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Exploration bonus
        """
        if action.visit_location is not None and action.visit_location not in state.visited:
            return self.exploration_bonus
        return 0.0

    def _calculate_progressive_reward(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate progressive reward.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Progressive reward
        """
        if action.visit_location is not None:
            # Progressive reward increases quadratically with number of visited locations
            visited_ratio = len(state.visited) / len(state.location_states)
            return self.progressive_factor * (visited_ratio ** 2)  # Quadratic scaling
        return 0.0

    def _calculate_belief_reward(
        self,
        belief: ISRSBelief,
        next_belief: ISRSBelief
    ) -> float:
        """Calculate belief-based reward.
        
        Args:
            belief: Current belief
            next_belief: Next belief
            
        Returns:
            Belief-based reward
        """
        # Calculate belief state similarity
        current_rmse, _ = belief.get_metrics(belief.sample_state(np.random.RandomState()))
        next_rmse, _ = next_belief.get_metrics(next_belief.sample_state(np.random.RandomState()))
        
        # Reward for reducing belief error
        belief_reward = (current_rmse - next_rmse) / max(1.0, current_rmse)
        return max(0.0, belief_reward) 