"""GP-MCTS policy for the Rover environment."""

from typing import Tuple, List, Dict, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from loguru import logger

from ..core.states import RoverBelief, RoverPos
from ..core.actions import RoverAction, RoverActionType
from ..env.rover_env import RoverEnv
from .base import BasePolicy

class MCTSNode:
    """Node in the MCTS tree with DPW support."""
    
    def __init__(self, state: np.ndarray, parent: Optional['MCTSNode'] = None):
        """Initialize MCTS node.
        
        Args:
            state: State vector
            parent: Parent node
        """
        self.state = state
        self.parent = parent
        self.children: Dict[Tuple[float, float], MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.reward = 0.0
        self.action_visits: Dict[Tuple[float, float], int] = {}
        self.observation_visits: Dict[Tuple[float, float], int] = {}
        
    def get_action_width(self, k_action: float, alpha_action: float) -> int:
        """Get the action width based on visit count."""
        return max(1, int(k_action * (self.visits ** alpha_action)))
        
    def get_observation_width(self) -> int:
        """Get the observation width based on visit count."""
        k_obs = 2.0  # Further increased for more observations
        alpha_obs = 0.8  # More aggressive widening
        return max(1, int(k_obs * (self.visits ** alpha_obs)))

class GPMCTSPolicy(BasePolicy):
    """GP-MCTS policy for rover exploration with mutual information rewards and DPW."""
    
    def __init__(
        self,
        env: RoverEnv,
        max_depth: int = 15,
        num_sims: int = 200,
        discount_factor: float = 0.95,
        kernel_params: Optional[dict] = None,
        k_action: float = 10.0,
        alpha_action: float = 0.8,
        k_obs: float = 10.0,
        alpha_obs: float = 0.8,
        exploration_constant: float = 3.0
    ):
        """Initialize GP-MCTS policy with DPW.
        
        Args:
            env: Rover environment
            max_depth: Maximum depth for MCTS search
            num_sims: Number of MCTS simulations
            discount_factor: Discount factor for rewards
            kernel_params: Parameters for the GP kernel
            k_action: Action widening constant
            alpha_action: Action widening exponent
            k_obs: Observation widening constant
            alpha_obs: Observation widening exponent
            exploration_constant: UCB exploration constant
        """
        super().__init__(env)
        self.grid_size = env.grid_size
        self.max_depth = max_depth
        self.num_sims = num_sims
        self.discount_factor = discount_factor
        self.k_action = k_action
        self.alpha_action = alpha_action
        self.k_obs = k_obs
        self.alpha_obs = alpha_obs
        self.exploration_constant = exploration_constant
        
        # Calculate state vector size
        width, height = self.grid_size
        grid_elements = width * height
        self.state_size = 2 + 3 * grid_elements
        
        # Initialize GP with tuned parameters matching RoverLocationBelief
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e1)) + \
                WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.1))
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # Initialize action patterns for better exploration
        self.action_patterns = self._generate_action_patterns()
        
    def _generate_action_patterns(self) -> List[Tuple[float, float]]:
        """Generate fixed action patterns for better exploration.
        
        Returns:
            List of action tuples (dx, dy)
        """
        patterns = []
        
        # Cardinal directions
        patterns.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
        
        # Diagonal directions
        patterns.extend([(1, 1), (-1, 1), (1, -1), (-1, -1)])
        
        # Longer moves
        for i in range(8):
            angle = 2 * np.pi * i / 8
            patterns.append((2 * np.cos(angle), 2 * np.sin(angle)))
            
        return patterns
        
    def _generate_actions(self, node: MCTSNode) -> List[Tuple[float, float]]:
        """Generate actions based on DPW and patterns.
        
        Args:
            node: Current node
            
        Returns:
            List of action tuples (dx, dy)
        """
        num_actions = node.get_action_width(self.k_action, self.alpha_action)
        actions = []
        
        # Add pattern-based actions first
        actions.extend(self.action_patterns)
        
        # Add random actions to reach desired width
        remaining = max(0, num_actions - len(actions))
        for _ in range(remaining):
            angle = np.random.uniform(0, 2 * np.pi)
            magnitude = np.random.uniform(0.5, 2.0)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            actions.append((dx, dy))
            
        return actions
        
    def _ucb_value(self, node: MCTSNode, child: MCTSNode) -> float:
        """Calculate UCB value for a child node with progressive widening.
        
        Args:
            node: Parent node
            child: Child node
            
        Returns:
            UCB value
        """
        if child.visits == 0:
            return float('inf')
            
        # Progressive widening factor
        pw_factor = (node.visits ** self.alpha_action) / (child.visits ** 0.5)
        
        # Standard UCB terms
        exploit = child.value / child.visits
        explore = self.exploration_constant * np.sqrt(np.log(node.visits) / child.visits)
        
        # Combine with progressive widening
        return exploit + explore * pw_factor
        
    def _select_action(self, node: MCTSNode) -> Tuple[float, float]:
        """Select action using UCB1 with DPW.
        
        Args:
            node: Current node
            
        Returns:
            Selected action as (dx, dy)
        """
        # Generate actions based on DPW
        actions = self._generate_actions(node)
        
        # If no actions are available, return a random action
        if not actions:
            angle = np.random.uniform(0, 2 * np.pi)
            return (np.cos(angle), np.sin(angle))
        
        # Calculate UCB values
        ucb_values = []
        for action in actions:
            if action in node.children:
                ucb = self._ucb_value(node, node.children[action])
            else:
                ucb = float('inf')  # Unexplored actions have highest priority
            ucb_values.append(ucb)
            
        # Select action with highest UCB value
        best_idx = np.argmax(ucb_values)
        return actions[best_idx]
        
    def _simulate(self, node: MCTSNode, depth: int) -> float:
        """Run a single MCTS simulation with DPW.
        
        Args:
            node: Current node
            depth: Current depth
            
        Returns:
            Simulated reward
        """
        if depth >= self.max_depth:
            return 0.0
            
        if node.visits == 0:
            # Leaf node - evaluate with GP
            reward = self._evaluate_state(node.state)
            node.value = reward
            node.visits = 1
            return reward
            
        # Select action using UCB with DPW
        action = self._select_action(node)
        
        # Create child node if it doesn't exist
        if action not in node.children:
            next_state = self._get_next_state(node.state, action)
            node.children[action] = MCTSNode(next_state, node)
            
        # Recursively simulate
        reward = self._simulate(node.children[action], depth + 1)
        reward = reward * self.discount_factor + self._evaluate_state(node.state)
        node.value += reward
        node.visits += 1
        
        return reward
        
    def get_action(self, belief: RoverBelief) -> RoverAction:
        """Get action using GP-MCTS.
        
        Args:
            belief: Current belief state
            
        Returns:
            Action to take
        """
        try:
            # Convert belief to state vector
            state = self._belief_to_state_vector(belief)
            
            # Run MCTS
            root = MCTSNode(state=state)
            for _ in range(self.num_sims):
                self._simulate(root, depth=0)
                
            # Select best action
            if not root.children:
                return self._get_random_action(belief)
                
            best_action = max(
                root.children.keys(),
                key=lambda a: root.children[a].value / root.children[a].visits
                if root.children[a].visits > 0 else float('-inf')
            )
            
            # Convert action to RoverAction
            dx, dy = best_action
            action_type = self._get_action_type(dx, dy)
            target_pos = RoverPos(
                x=max(0, min(self.grid_size[0] - 1, belief.pos.x + int(dx))),
                y=max(0, min(self.grid_size[1] - 1, belief.pos.y + int(dy)))
            )
            
            return RoverAction(action_type=action_type, target_pos=target_pos)
            
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
        pass  # GP-MCTS doesn't learn from experience
        
    def _get_random_action(self, belief: RoverBelief) -> RoverAction:
        """Get random action.
        
        Args:
            belief: Current belief state
            
        Returns:
            Random action
        """
        action_types = [
            RoverActionType.UP, RoverActionType.DOWN,
            RoverActionType.LEFT, RoverActionType.RIGHT,
            RoverActionType.NE, RoverActionType.NW,
            RoverActionType.SE, RoverActionType.SW
        ]
        action_type = self.rng.choice(action_types)
        
        # Get direction vector
        dx, dy = {
            RoverActionType.UP: (0, 1),
            RoverActionType.DOWN: (0, -1),
            RoverActionType.LEFT: (-1, 0),
            RoverActionType.RIGHT: (1, 0),
            RoverActionType.NE: (1, 1),
            RoverActionType.NW: (-1, 1),
            RoverActionType.SE: (1, -1),
            RoverActionType.SW: (-1, -1)
        }[action_type]
        
        target_pos = RoverPos(
            x=max(0, min(self.grid_size[0] - 1, belief.pos.x + dx)),
            y=max(0, min(self.grid_size[1] - 1, belief.pos.y + dy))
        )
        
        return RoverAction(action_type=action_type, target_pos=target_pos)
        
    def _get_action_type(self, dx: float, dy: float) -> RoverActionType:
        """Convert action vector to RoverActionType.
        
        Args:
            dx: x component of action
            dy: y component of action
            
        Returns:
            RoverActionType
        """
        dx = int(dx)
        dy = int(dy)
        
        if dx == 0 and dy == 1:
            return RoverActionType.UP
        elif dx == 0 and dy == -1:
            return RoverActionType.DOWN
        elif dx == -1 and dy == 0:
            return RoverActionType.LEFT
        elif dx == 1 and dy == 0:
            return RoverActionType.RIGHT
        elif dx == 1 and dy == 1:
            return RoverActionType.NE
        elif dx == -1 and dy == 1:
            return RoverActionType.NW
        elif dx == 1 and dy == -1:
            return RoverActionType.SE
        elif dx == -1 and dy == -1:
            return RoverActionType.SW
        else:
            return RoverActionType.WAIT
        
    def _validate_state(self, state: np.ndarray) -> None:
        """Validate state vector dimensions.
        
        Args:
            state: State vector to validate
            
        Raises:
            ValueError: If state dimensions are incorrect
        """
        if state.shape[0] != self.state_size:
            raise ValueError(
                f"State vector size {state.shape[0]} does not match "
                f"expected size {self.state_size}"
            )
            
    def _evaluate_state(self, state: np.ndarray) -> float:
        """Evaluate state using enhanced reward function.
        
        Args:
            state: State vector containing position, belief map, uncertainty map,
                  and visited positions mask
            
        Returns:
            Reward value for the state
        """
        self._validate_state(state)
        
        # Extract components from state vector
        width, height = self.grid_size
        grid_size = width * height
        
        belief_map = state[2:2+grid_size].reshape(height, width)
        uncertainty_map = state[2+grid_size:2+2*grid_size].reshape(height, width)
        visited_mask = state[2+2*grid_size:].reshape(height, width)
        
        # 1. Mutual Information Reward
        unvisited_mask = 1 - visited_mask
        uncertainty_score = np.sum(uncertainty_map * unvisited_mask)
        mutual_info = 20.0 * uncertainty_score * (1 + np.sum(unvisited_mask) / grid_size)
        
        # 2. Novelty Reward
        new_cells = np.sum(unvisited_mask) / grid_size
        novelty = new_cells * 15.0
        
        # 3. Distance Penalty
        pos_x, pos_y = state[:2]
        dist_to_center = np.sqrt((pos_x - width/2)**2 + (pos_y - height/2)**2)
        dist_penalty = 0.005 * dist_to_center
        
        # 4. Frontier Reward with Progressive Scaling
        frontier_mask = np.zeros_like(visited_mask)
        frontier_cells = 0
        for i in range(1, height-1):
            for j in range(1, width-1):
                if visited_mask[i,j] == 1:
                    # Check 8-connected neighborhood
                    neighborhood = visited_mask[i-1:i+2, j-1:j+2]
                    if np.any(neighborhood == 0):  # Has unvisited neighbor
                        frontier_mask[i,j] = 1
                        frontier_cells += 1
        
        # Scale frontier reward based on exploration progress
        exploration_progress = np.sum(visited_mask) / grid_size
        frontier_scale = 10.0 * (1 - exploration_progress)  # Higher reward early in exploration
        frontier = frontier_cells * frontier_scale
        
        # 5. Exploration Bonus with Progressive Scaling
        if exploration_progress < 0.7:  # Only apply during early and mid exploration
            exploration_bonus = 20.0 * (1 - exploration_progress) * uncertainty_score
        else:
            exploration_bonus = 0.0
        
        # Combine rewards with dynamic weighting
        total_reward = mutual_info + novelty + frontier + exploration_bonus - dist_penalty
        
        return total_reward
        
    def _get_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Get next state after taking action.
        
        Args:
            state: Current state vector
            action: Action vector [dx, dy]
            
        Returns:
            Next state vector
        """
        self._validate_state(state)
        
        next_state = state.copy()
        
        # Update position
        width, height = self.grid_size
        pos_x, pos_y = state[:2]
        dx, dy = action
        
        # Clip to grid bounds
        next_x = np.clip(pos_x + dx, 0, width - 1)
        next_y = np.clip(pos_y + dy, 0, height - 1)
        
        next_state[0] = next_x
        next_state[1] = next_y
        
        # Mark new position as visited
        grid_size = width * height
        visited_idx = 2 + 2*grid_size + int(next_y * width + next_x)
        next_state[visited_idx] = 1
        
        return next_state 