"""POMCP implementation for ISRS environment."""

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from .actions import MultimodalIPPAction
from .states import ISRSWorldState, ISRSObservation
from .belief import ISRSBelief
from .reward_calculator import RewardCalculator

@dataclass
class POMCPNode:
    """Node in the POMCP search tree."""
    action: Optional[MultimodalIPPAction]  # Action that led to this node
    parent: Optional['POMCPNode']  # Parent node
    children: Dict[MultimodalIPPAction, 'POMCPNode']  # Child nodes
    visits: int  # Number of times this node has been visited
    value: float  # Value estimate for this node
    belief: ISRSBelief  # Belief state at this node
    observation_counts: Dict[ISRSObservation, int]  # Counts of observations

    def __init__(
        self,
        action: Optional[MultimodalIPPAction] = None,
        parent: Optional['POMCPNode'] = None,
        belief: Optional[ISRSBelief] = None
    ) -> None:
        """Initialize POMCP node.
        
        Args:
            action: Action that led to this node
            parent: Parent node
            belief: Belief state at this node
        """
        self.action = action
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.belief = belief
        self.observation_counts = defaultdict(int)

    def is_leaf(self) -> bool:
        """Check if node is a leaf node.
        
        Returns:
            True if node is a leaf node
        """
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if node is the root node.
        
        Returns:
            True if node is the root node
        """
        return self.parent is None

    def get_ucb(self, exploration_constant: float) -> float:
        """Calculate UCB value for node.
        
        Args:
            exploration_constant: UCB exploration constant
            
        Returns:
            UCB value
        """
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_constant * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

class POMCP:
    """POMCP implementation for ISRS environment."""
    
    def __init__(
        self,
        env,
        reward_calculator: RewardCalculator,
        exploration_constant: float = 1.0,
        discount_factor: float = 0.95,
        max_depth: int = 50,
        num_simulations: int = 1000,
        rng: np.random.RandomState = None
    ) -> None:
        """Initialize POMCP.
        
        Args:
            env: Environment instance
            reward_calculator: Reward calculator instance
            exploration_constant: UCB exploration constant
            discount_factor: Discount factor for future rewards
            max_depth: Maximum search depth
            num_simulations: Number of simulations per action
            rng: Random number generator
        """
        self.env = env
        self.reward_calculator = reward_calculator
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.rng = rng or np.random.RandomState(42)
        self.root = None

    def search(
        self,
        belief: ISRSBelief,
        available_actions: List[MultimodalIPPAction]
    ) -> MultimodalIPPAction:
        """Search for best action using POMCP.
        
        Args:
            belief: Current belief state
            available_actions: List of available actions
            
        Returns:
            Best action to take
        """
        # Initialize root node
        self.root = POMCPNode(belief=belief)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Sample a state from belief
            state = belief.sample_state(self.rng)
            
            # Run simulation
            self._simulate(state, self.root, 0)
            
        # Select best action
        best_action = max(
            available_actions,
            key=lambda a: self.root.children[a].value / self.root.children[a].visits
        )
        
        return best_action

    def _simulate(
        self,
        state: ISRSWorldState,
        node: POMCPNode,
        depth: int
    ) -> float:
        """Run a simulation from current state and node.
        
        Args:
            state: Current state
            node: Current node
            depth: Current depth
            
        Returns:
            Value estimate
        """
        # Check if we've reached max depth
        if depth >= self.max_depth:
            return 0.0
            
        # Check if node is a leaf
        if node.is_leaf():
            # Expand node if not terminal
            if not self._is_terminal(state):
                self._expand(node)
                return self._rollout(state, depth)
            return 0.0
            
        # Select action using UCB
        action = self._select_action(node)
        
        # Generate next state and observation
        next_state, observation, reward, done, next_belief = self.env.step(action)
        
        # Find or create child node
        if observation not in node.observation_counts:
            child = POMCPNode(action=action, parent=node, belief=next_belief)
            node.children[action] = child
        else:
            child = node.children[action]
            
        # Update observation counts
        node.observation_counts[observation] += 1
        
        # Recursively simulate
        value = reward + self.discount_factor * self._simulate(
            next_state,
            child,
            depth + 1
        )
        
        # Update node statistics
        node.visits += 1
        node.value += value
        
        return value

    def _expand(self, node: POMCPNode) -> None:
        """Expand a leaf node.
        
        Args:
            node: Node to expand
        """
        # Get available actions
        available_actions = self._get_available_actions(node.belief)
        
        # Create child nodes for each action
        for action in available_actions:
            node.children[action] = POMCPNode(action=action, parent=node)

    def _rollout(self, state: ISRSWorldState, depth: int) -> float:
        """Run a rollout from current state.
        
        Args:
            state: Current state
            depth: Current depth
            
        Returns:
            Value estimate
        """
        if depth >= self.max_depth:
            return 0.0
            
        # Get available actions
        available_actions = self._get_available_actions(
            ISRSBelief(
                current=state.current,
                visited=state.visited,
                location_beliefs=[],  # Empty beliefs for rollout
                cost_expended=state.cost_expended
            )
        )
        
        # Select random action
        action = self.rng.choice(available_actions)
        
        # Generate next state and observation
        next_state, _, reward, done, _ = self.env.step(action)
        
        # Recursively rollout
        return reward + self.discount_factor * self._rollout(next_state, depth + 1)

    def _select_action(self, node: POMCPNode) -> MultimodalIPPAction:
        """Select action using UCB.
        
        Args:
            node: Current node
            
        Returns:
            Selected action
        """
        return max(
            node.children.keys(),
            key=lambda a: node.children[a].get_ucb(self.exploration_constant)
        )

    def _get_available_actions(self, belief: ISRSBelief) -> List[MultimodalIPPAction]:
        """Get list of available actions.
        
        Args:
            belief: Current belief state
            
        Returns:
            List of available actions
        """
        available_actions = []
        
        # Add visit actions for unvisited locations
        for i in range(self.env.n_locations):
            if i not in belief.visited:
                available_actions.append(
                    MultimodalIPPAction(
                        visit_location=i,
                        sensing_action=None,
                        cost=1.0
                    )
                )
                
        # Add sensing action for current location
        available_actions.append(
            MultimodalIPPAction(
                visit_location=None,
                sensing_action=ISRSSensor(efficiency=0.8),
                cost=0.1
            )
        )
        
        return available_actions

    def _is_terminal(self, state: ISRSWorldState) -> bool:
        """Check if state is terminal.
        
        Args:
            state: Current state
            
        Returns:
            True if state is terminal
        """
        return len(state.visited) == self.env.n_locations 