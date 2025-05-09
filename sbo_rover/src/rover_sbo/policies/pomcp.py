"""POMCP (Partially Observable Monte Carlo Planning) policy implementation for the Rover environment."""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from loguru import logger

from ..core.states import (
    RoverWorldState, RoverBelief, RoverPos
)
from ..core.actions import RoverAction, RoverActionType
from ..env.rover_env import RoverEnv
from .base import BasePolicy

# Define movement directions
DIRECTIONS = {
    RoverActionType.UP: (0, 1),
    RoverActionType.DOWN: (0, -1),
    RoverActionType.LEFT: (-1, 0),
    RoverActionType.RIGHT: (1, 0),
    RoverActionType.NE: (1, 1),
    RoverActionType.NW: (-1, 1),
    RoverActionType.SE: (1, -1),
    RoverActionType.SW: (-1, -1)
}

@dataclass
class POMCPNode:
    """Node in the POMCP search tree."""
    action: Optional[RoverAction] = None
    visits: int = 0
    value: float = 0.0
    children: Dict[RoverAction, 'POMCPNode'] = None
    
    def __post_init__(self):
        """Initialize children dictionary."""
        if self.children is None:
            self.children = {}
            
    def ucb(self, parent_visits: int, exploration_constant: float) -> float:
        """Calculate UCB value for node selection.
        
        Args:
            parent_visits: Number of visits to parent node
            exploration_constant: Exploration constant
            
        Returns:
            UCB value
        """
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_constant * np.sqrt(np.log(parent_visits) / self.visits)
        
    def best_child(self, exploration_constant: float) -> 'POMCPNode':
        """Get best child node according to UCB.
        
        Args:
            exploration_constant: Exploration constant
            
        Returns:
            Best child node
        """
        if not self.children:
            return None
        return max(
            self.children.values(),
            key=lambda c: c.ucb(self.visits, exploration_constant)
        )

class POMCPPolicy(BasePolicy):
    """POMCP policy implementation."""
    
    def __init__(
        self,
        env: RoverEnv,
        num_particles: int = 100,
        max_depth: int = 15,
        num_sims: int = 200,
        discount_factor: float = 0.95,
        exploration_constant: float = 1.0,
        **kwargs
    ):
        """Initialize POMCP policy.
        
        Args:
            env: Environment instance
            num_particles: Number of particles for belief state
            max_depth: Maximum tree depth
            num_sims: Number of simulations per action selection
            discount_factor: Discount factor for future rewards
            exploration_constant: UCB exploration constant
        """
        super().__init__(env)
        self.num_particles = num_particles
        self.max_depth = max_depth
        self.num_sims = num_sims
        self.discount_factor = discount_factor
        self.exploration_constant = exploration_constant
        self.grid_size = env.grid_size
        
    def get_action(self, belief: RoverBelief) -> RoverAction:
        """Get action using POMCP.
        
        Args:
            belief: Current belief state
            
        Returns:
            Action to take
        """
        try:
            if belief is None or belief.location_beliefs is None:
                logger.warning("Belief state is None, using random action")
                return self._get_random_action(belief)
                
            # Run POMCP simulations
            root = POMCPNode()
            for _ in range(self.num_sims):
                # Sample state from belief
                state = self._sample_state(belief)
                if state is None:
                    continue
                    
                # Simulate
                self._simulate(root, state, depth=0)
                
            # Select best action
            if not root.children:
                return self._get_random_action(belief)
                
            best_action = max(
                root.children.keys(),
                key=lambda a: root.children[a].value / root.children[a].visits
                if root.children[a].visits > 0 else float('-inf')
            )
            
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
        pass  # POMCP doesn't learn from experience
        
    def _simulate(self, node: POMCPNode, state: RoverWorldState, depth: int) -> float:
        """Run simulation from given node and state.
        
        Args:
            node: Current node
            state: Current state
            depth: Current depth
            
        Returns:
            Total discounted reward
        """
        # Check if max depth reached
        if depth >= self.max_depth:
            return 0.0
            
        # Check if node is leaf
        if not node.children:
            # Expand node
            for action in self._get_actions(state):
                node.children[action] = POMCPNode(action=action)
                
        # Select action using UCB
        action_node = node.best_child(self.exploration_constant)
        if action_node is None:
            return 0.0
            
        # Take action and get next state
        next_state = self.env.generate_s(state, action_node.action)
        reward = self.env._calculate_reward(state, next_state)
        done = (next_state.cost_expended >= self.env.cost_budget or
                next_state.pos == self.env.goal_pos)
        
        # Update node statistics
        node.visits += 1
        node.value += reward
        
        # Recursively simulate
        if not done:
            reward += self.discount_factor * self._simulate(
                action_node, next_state, depth + 1
            )
            
        return reward
        
    def _sample_state(self, belief: RoverBelief) -> Optional[RoverWorldState]:
        """Sample state from belief.
        
        Args:
            belief: Current belief state
            
        Returns:
            Sampled state or None if sampling fails
        """
        try:
            if belief is None or belief.location_beliefs is None:
                return None
                
            # Sample rover position
            pos = belief.pos
            
            # Sample location states for each location
            location_states = []
            for loc_belief in belief.location_beliefs:
                sampled_type = self.rng.choice(len(loc_belief.probs), p=loc_belief.probs)
                location_states.append(sampled_type)
            location_states = np.array(location_states)
            
            # Other fields as needed
            return RoverWorldState(
                pos=pos,
                visited=belief.visited,
                location_states=location_states,
                cost_expended=belief.cost_expended,
                drill_samples=belief.drill_samples
            )
        except Exception as e:
            logger.error(f"Error in _sample_state: {str(e)}")
            return None
        
    def _get_actions(self, state: RoverWorldState) -> List[RoverAction]:
        """Get available actions for state.
        
        Args:
            state: Current state
            
        Returns:
            List of available actions
        """
        actions = []
        x, y = state.pos.x, state.pos.y
        # Add movement actions with correct target_pos as RoverPos
        if x > 0:
            actions.append(RoverAction(RoverActionType.LEFT, target_pos=RoverPos(x-1, y)))
        if x < self.env.map_size[0] - 1:
            actions.append(RoverAction(RoverActionType.RIGHT, target_pos=RoverPos(x+1, y)))
        if y > 0:
            actions.append(RoverAction(RoverActionType.DOWN, target_pos=RoverPos(x, y-1)))
        if y < self.env.map_size[1] - 1:
            actions.append(RoverAction(RoverActionType.UP, target_pos=RoverPos(x, y+1)))
        # Add drill action if at location
        loc_idx = self.env._pos_to_idx(state.pos)
        if state.location_states[loc_idx] == 0:  # Empty
            actions.append(RoverAction(RoverActionType.DRILL, target_pos=RoverPos(x, y)))
        return actions
        
    def _get_random_action(self, belief: Optional[RoverBelief]) -> RoverAction:
        """Get random action.
        
        Args:
            belief: Current belief state (can be None)
            
        Returns:
            Random action
        """
        try:
            # Get valid actions
            if belief is None or belief.pos is None:
                # If no belief or position, just move randomly
                action_type = self.rng.choice([
                    RoverActionType.UP, RoverActionType.DOWN,
                    RoverActionType.LEFT, RoverActionType.RIGHT,
                    RoverActionType.NE, RoverActionType.NW,
                    RoverActionType.SE, RoverActionType.SW
                ])
                # Get direction vector for the action
                dx, dy = DIRECTIONS[action_type]
                target_pos = RoverPos(
                    x=max(0, min(self.grid_size[0] - 1, dx)),
                    y=max(0, min(self.grid_size[1] - 1, dy))
                )
            else:
                # Choose between movement and drill
                if self.rng.random() < 0.8:  # 80% chance to move
                    action_type = self.rng.choice([
                        RoverActionType.UP, RoverActionType.DOWN,
                        RoverActionType.LEFT, RoverActionType.RIGHT,
                        RoverActionType.NE, RoverActionType.NW,
                        RoverActionType.SE, RoverActionType.SW
                    ])
                    # Get direction vector for the action
                    dx, dy = DIRECTIONS[action_type]
                    target_pos = RoverPos(
                        x=max(0, min(self.grid_size[0] - 1, belief.pos.x + dx)),
                        y=max(0, min(self.grid_size[1] - 1, belief.pos.y + dy))
                    )
                else:
                    # Drill at current position
                    action_type = RoverActionType.DRILL
                    target_pos = belief.pos
                
            return RoverAction(
                action_type=action_type,
                target_pos=target_pos
            )
        except Exception as e:
            logger.error(f"Error in _get_random_action: {str(e)}")
            # Return a safe default action
            return RoverAction(
                action_type=RoverActionType.WAIT,
                target_pos=RoverPos(0, 0)
            )