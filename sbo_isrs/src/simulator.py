"""Simulation code for ISRS environment."""

from typing import List, Optional
import numpy as np
from dataclasses import dataclass
from loguru import logger

# Local imports
from .env import ISRSEnv
from .states import ISRSWorldState, ISRSBelief
from .belief import ISRSBelief
from .policies import (
    BasePolicy, InformationSeekingPolicy
)
from .actions import MultimodalIPPAction, ISRSSensor

@dataclass
class SimulationResult:
    """Results from a single simulation episode.
    
    Attributes:
        total_reward: Total reward accumulated during episode
        steps_taken: Number of steps taken
        visited_locations: Set of visited locations
        belief_errors: List of belief errors (RMSE) at each step
        info_gains: List of information gains at each step
    """
    total_reward: float
    steps_taken: int
    visited_locations: set
    belief_errors: List[float]
    info_gains: List[float]

class Simulator:
    """Simulator for running ISRS environment episodes."""
    
    def __init__(
        self,
        env: ISRSEnv,
        policy: BasePolicy,
        max_steps: int = 100,
        seed: Optional[int] = None
    ) -> None:
        """Initialize simulator.
        
        Args:
            env: Environment instance
            policy: Policy to use for action selection
            max_steps: Maximum number of steps per episode
            seed: Random seed for reproducibility
        """
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        
    def run_episode(self) -> SimulationResult:
        """Run a single episode.
        
        Returns:
            Simulation results
        """
        # Reset environment
        state, observation, belief = self.env.reset()
        
        # Initialize tracking variables
        total_reward = 0.0
        steps_taken = 0
        visited_locations = set()
        belief_errors = []
        info_gains = []
        
        # Run episode
        while steps_taken < self.max_steps:
            # Get available actions
            available_actions = self._get_available_actions(state)
            
            # Select action using policy
            action = self.policy.select_action(belief, available_actions)
            
            # Execute action
            next_state, next_observation, reward, done, next_belief = self.env.step(action)
            
            # Update tracking variables
            total_reward += reward
            steps_taken += 1
            visited_locations.add(state.current)
            
            # Calculate belief error and information gain
            rmse, _ = belief.get_metrics(state)
            belief_errors.append(rmse)
            
            if isinstance(self.policy, InformationSeekingPolicy):
                info_gain = self._calculate_info_gain(belief, next_belief)
                info_gains.append(info_gain)
            
            # Update state and belief
            state = next_state
            observation = next_observation
            belief = next_belief
            
            # Check if episode is done
            if done:
                break
                
        return SimulationResult(
            total_reward=total_reward,
            steps_taken=steps_taken,
            visited_locations=visited_locations,
            belief_errors=belief_errors,
            info_gains=info_gains
        )
        
    def run_episodes(self, num_episodes: int) -> List[SimulationResult]:
        """Run multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            List of simulation results
        """
        results = []
        for i in range(num_episodes):
            logger.info(f"Running episode {i+1}/{num_episodes}")
            result = self.run_episode()
            results.append(result)
            logger.info(f"Episode {i+1} complete: reward={result.total_reward:.2f}, steps={result.steps_taken}")
        return results
        
    def _get_available_actions(self, state: ISRSWorldState) -> List[MultimodalIPPAction]:
        """Get list of available actions.
        
        Args:
            state: Current state
            
        Returns:
            List of available actions
        """
        actions = []
        
        # Add visit actions for unvisited locations
        for i in range(self.env.num_locations):
            if i not in state.visited:
                # Calculate cost based on distance
                cost = self.env.shortest_paths[state.current, i]
                actions.append(MultimodalIPPAction(
                    visit_location=i,
                    sensing_action=None,
                    cost=cost
                ))
        
        # Add sensing action for current location
        actions.append(MultimodalIPPAction(
            visit_location=None,
            sensing_action=ISRSSensor.MEDIUM,  # Use medium efficiency sensor
            cost=0.1  # Fixed cost for sensing
        ))
        
        return actions
        
    def _calculate_info_gain(self, current_belief: ISRSBelief, next_belief: ISRSBelief) -> float:
        """Calculate information gain between belief states.
        
        Args:
            current_belief: Current belief state
            next_belief: Next belief state
            
        Returns:
            Information gain
        """
        # Sample states for comparison
        current_state = current_belief.sample_state(self.rng)
        next_state = next_belief.sample_state(self.rng)
        
        # Calculate belief metrics
        current_rmse, current_trace = current_belief.get_metrics(current_state)
        next_rmse, next_trace = next_belief.get_metrics(next_state)
        
        # Calculate information gain as reduction in uncertainty
        info_gain = (current_trace - next_trace) / max(1.0, current_trace)
        
        return info_gain

def run_simulation(
    env: ISRSEnv,
    policy: BasePolicy,
    num_episodes: int,
    max_steps: int,
    seed: Optional[int] = None
) -> SimulationResult:
    """Run simulation with specified policy.
    
    Args:
        env: Environment instance
        policy: Policy to use
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        seed: Random seed
        
    Returns:
        Simulation results
    """
    # Create simulator
    simulator = Simulator(
        env=env,
        policy=policy,
        max_steps=max_steps,
        seed=seed
    )
    
    # Run episodes
    return simulator.run_episodes(num_episodes) 