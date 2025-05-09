"""Base policy class for the Rover environment."""

from abc import ABC, abstractmethod
import numpy as np

from ..core.states import RoverBelief
from ..core.actions import RoverAction
from ..env.rover_env import RoverEnv

class BasePolicy(ABC):
    """Base class for all policies."""
    
    def __init__(self, env: RoverEnv):
        """Initialize base policy.
        
        Args:
            env: Rover environment
        """
        self.env = env
        self.rng = np.random.RandomState()
    
    @abstractmethod
    def get_action(self, belief: RoverBelief) -> RoverAction:
        """Get action from policy.
        
        Args:
            belief: Current belief state
            
        Returns:
            Action to take
        """
        pass
        
    @abstractmethod
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
        pass 