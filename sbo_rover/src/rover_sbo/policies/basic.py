"""Basic policy implementation for the Rover environment."""

from typing import List
import numpy as np
from loguru import logger

from ..core.states import (
    RoverBelief, RoverPos
)
from ..core.actions import RoverAction, RoverActionType
from ..env.rover_env import RoverEnv
from .base import BasePolicy

class BasicPolicy(BasePolicy):
    """Basic policy implementation."""
    
    def __init__(
        self,
        env: RoverEnv,
        rng: np.random.RandomState,
        exploration_prob: float = 0.1
    ):
        """Initialize basic policy.
        
        Args:
            env: Rover environment
            rng: Random number generator
            exploration_prob: Probability of random exploration
        """
        self.env = env
        self.rng = rng
        self.exploration_prob = exploration_prob
        
    def get_action(self, belief: RoverBelief) -> RoverAction:
        """Get action from policy.
        
        Args:
            belief: Current belief state
            
        Returns:
            Action to take
        """
        try:
            # Random exploration
            if self.rng.random() < self.exploration_prob:
                return self._get_random_action(belief)
                
            # Get available actions
            actions = self._get_available_actions(belief)
            
            # Choose action with highest expected value
            best_action = None
            best_value = float('-inf')
            for action in actions:
                value = self._evaluate_action(belief, action)
                if value > best_value:
                    best_value = value
                    best_action = action
                    
            if best_action is None:
                return self._get_random_action(belief)
                
            return best_action
            
        except Exception as e:
            logger.error(f"Error in get_action: {str(e)}")
            return self._get_random_action(belief)
            
    def update(self, state: RoverBelief, action: RoverAction, reward: float,
              next_state: RoverBelief, done: bool) -> None:
        """Update policy using experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass  # Basic policy doesn't learn
        
    def _get_random_action(self, belief: RoverBelief) -> RoverAction:
        """Get random action.
        
        Args:
            belief: Current belief state
            
        Returns:
            Random action
        """
        actions = self._get_available_actions(belief)
        return self.rng.choice(actions)
        
    def _get_available_actions(self, belief: RoverBelief) -> List[RoverAction]:
        """Get list of available actions.
        
        Args:
            belief: Current belief state
            
        Returns:
            List of available actions
        """
        actions = []
        
        # Movement actions
        if belief.pos.y < self.env.map_size[1] - 1:
            actions.append(RoverAction(RoverActionType.UP))
        if belief.pos.y > 0:
            actions.append(RoverAction(RoverActionType.DOWN))
        if belief.pos.x > 0:
            actions.append(RoverAction(RoverActionType.LEFT))
        if belief.pos.x < self.env.map_size[0] - 1:
            actions.append(RoverAction(RoverActionType.RIGHT))
            
        # Drill action (can drill at any location)
        for i in range(self.env.map_size[0]):
            for j in range(self.env.map_size[1]):
                actions.append(RoverAction(RoverActionType.DRILL, (i, j)))
                
        return actions
        
    def _evaluate_action(self, belief: RoverBelief, action: RoverAction) -> float:
        """Evaluate action.
        
        Args:
            belief: Current belief state
            action: Action to evaluate
            
        Returns:
            Action value
        """
        if action.action_type == RoverActionType.DRILL:
            # Evaluate drill action
            pos = RoverPos(*action.target_pos)
            if pos in belief.visited:
                return float('-inf')  # Don't drill at visited locations
                
            idx = self.env._pos_to_idx(pos)
            if belief.location_beliefs[idx].gp_mean is not None:
                return belief.location_beliefs[idx].gp_mean
            return 0.0
            
        else:
            # Evaluate movement action
            next_pos = self.env._get_next_pos(belief.pos, action)
            if next_pos in belief.visited:
                return -1.0  # Penalize revisiting locations
            return 0.0 