"""Enhanced GP-MCTS policy with adaptive parameters and improved exploration."""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np

from .gp_mcts import GPMCTSPolicy
from ..core.states import RoverBelief, RoverPos
from ..core.actions import RoverAction, RoverActionType
from ..env.rover_env import RoverEnv

# Direction vectors for each action type
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

class MCTSNode:
    """Node in MCTS tree."""
    
    def __init__(self, state: Optional[np.ndarray] = None, action: Optional[RoverAction] = None):
        """Initialize node.
        
        Args:
            state: State vector
            action: Action that led to this node
        """
        self.state = state
        self.action = action
        self.value = 0.0
        self.visits = 0
        self.children: Dict[RoverAction, MCTSNode] = {}

class EnhancedGPMCTSPolicy(GPMCTSPolicy):
    """Enhanced GP-MCTS policy with adaptive parameters and improved exploration strategies."""
    
    def __init__(
        self,
        env: RoverEnv,
        max_depth: int = 15,
        num_sims: int = 200,
        discount_factor: float = 0.95,
        kernel_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize Enhanced GP-MCTS policy.
        
        Args:
            env: Rover environment
            max_depth: Maximum tree depth
            num_sims: Number of simulations per action selection
            discount_factor: Discount factor for future rewards
            kernel_params: Kernel parameters for GP
        """
        super().__init__(env, max_depth, num_sims, discount_factor, kernel_params)
        
    def get_action(self, belief: RoverBelief) -> RoverAction:
        """Get action using enhanced GP-MCTS.
        
        Args:
            belief: Current belief state
            
        Returns:
            Action to take
        """
        # Convert belief to state vector
        state = self._belief_to_state_vector(belief)
        
        # Run MCTS
        root = MCTSNode(state=state)
        for _ in range(self.num_sims):
            self._simulate(root, depth=0)
        
        # Select best action
        if not root.children:
            return self._get_random_action(belief)
        
        best_node = max(
            root.children.values(),
            key=lambda n: n.value / n.visits if n.visits > 0 else float('-inf')
        )
        
        return best_node.action
        
    def _simulate(self, node: MCTSNode, depth: int = 0) -> float:
        """Run simulation from given node.
        
        Args:
            node: Current node
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
            actions = self._generate_actions(node)
            for action in actions:
                next_state = self._get_next_state(node.state, action)
                child = MCTSNode(state=next_state, action=action)
                node.children[action] = child
                
        # Select action using UCB
        action = self._select_action(node)
        if action is None:
            return 0.0
            
        # Take action and get next state
        next_node = node.children[action]
        reward = self._evaluate_state(next_node.state)
        
        # Update node statistics
        node.visits += 1
        node.value += reward
        
        # Recursively simulate
        reward += self.discount_factor * self._simulate(next_node, depth + 1)
        
        return reward
        
    def _generate_actions(self, node: MCTSNode) -> List[RoverAction]:
        """Generate available actions for node.
        
        Args:
            node: Current node
            
        Returns:
            List of available actions
        """
        actions = []
        pos = self._state_vector_to_pos(node.state)
        
        # Add movement actions
        for action_type, (dx, dy) in DIRECTIONS.items():
            new_x = pos.x + dx
            new_y = pos.y + dy
            
            # Check bounds
            if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                actions.append(RoverAction(
                    action_type=action_type,
                    target_pos=RoverPos(x=new_x, y=new_y)
                ))
                
        # Add drill action
        actions.append(RoverAction(
            action_type=RoverActionType.DRILL,
            target_pos=pos
        ))
        
        return actions
        
    def _select_action(self, node: MCTSNode) -> Optional[RoverAction]:
        """Select action using UCB1.
        
        Args:
            node: Current node
            
        Returns:
            Selected action or None if no actions available
        """
        if not node.children:
            return None
            
        # Find action with highest UCB value
        best_value = float('-inf')
        best_action = None
        
        for action, child in node.children.items():
            if child.visits == 0:
                return action
                
            # Calculate UCB value
            exploit = child.value / child.visits
            explore = self.exploration_constant * np.sqrt(
                np.log(node.visits) / child.visits
            )
            ucb = exploit + explore
            
            if ucb > best_value:
                best_value = ucb
                best_action = action
                
        return best_action
        
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
        action_type = np.random.choice(action_types)
        
        # Get direction vector
        dx, dy = DIRECTIONS[action_type]
        target_pos = RoverPos(
            x=max(0, min(self.grid_size[0] - 1, belief.pos.x + dx)),
            y=max(0, min(self.grid_size[1] - 1, belief.pos.y + dy))
        )
        
        return RoverAction(action_type=action_type, target_pos=target_pos)
        
    def _belief_to_state_vector(self, belief: RoverBelief) -> np.ndarray:
        """Convert belief to state vector.
        
        Args:
            belief: Current belief state
            
        Returns:
            State vector
        """
        width, height = self.grid_size
        grid_size = width * height
        
        # Initialize state vector
        state = np.zeros(2 + 3 * grid_size)
        
        # Set rover position
        state[0] = belief.pos.x
        state[1] = belief.pos.y
        
        # Set belief map
        belief_map = np.zeros(grid_size)
        uncertainty_map = np.zeros(grid_size)
        visited_map = np.zeros(grid_size)
        
        for i, loc_belief in enumerate(belief.location_beliefs):
            if loc_belief.gp_mean is not None:
                belief_map[i] = loc_belief.gp_mean
                uncertainty_map[i] = loc_belief.gp_std
                
        for pos in belief.visited:
            idx = pos.y * width + pos.x
            visited_map[idx] = 1
            
        state[2:2+grid_size] = belief_map
        state[2+grid_size:2+2*grid_size] = uncertainty_map
        state[2+2*grid_size:] = visited_map
        
        return state
        
    def _state_vector_to_pos(self, state_vector: np.ndarray) -> RoverPos:
        """Convert state vector to RoverPos.
        
        Args:
            state_vector: State vector
            
        Returns:
            RoverPos
        """
        pos_x = int(state_vector[0])
        pos_y = int(state_vector[1])
        return RoverPos(x=pos_x, y=pos_y)
        
    def _get_next_state(self, state: np.ndarray, action: RoverAction) -> np.ndarray:
        """Get next state after taking action.
        
        Args:
            state: Current state vector
            action: Action to take
            
        Returns:
            Next state vector
        """
        if action.action_type == RoverActionType.DRILL:
            return state.copy()
            
        # Get direction from action type
        dx, dy = DIRECTIONS[action.action_type]
        
        # Update position
        pos = self._state_vector_to_pos(state)
        new_pos = RoverPos(x=pos.x + dx, y=pos.y + dy)
        
        # Convert back to state vector
        next_state = state.copy()
        next_state[0] = new_pos.x
        next_state[1] = new_pos.y
        
        return next_state

    def _update_action_memory(self, action: Tuple[float, float], reward: float) -> None:
        """Update memory of successful actions.
        
        Args:
            action: Action tuple (dx, dy)
            reward: Reward obtained from the action
        """
        if len(self.action_memory['successful']) >= self.memory_size:
            self.action_memory['successful'].pop(0)
            self.action_memory['weights'].pop(0)
        
        self.action_memory['successful'].append(action)
        self.action_memory['weights'].append(max(0.1, min(1.0, reward / 1000.0)))
        
        total_weight = sum(self.action_memory['weights'])
        if total_weight > 0:
            self.action_memory['weights'] = [w/total_weight for w in self.action_memory['weights']]
            
    def _get_current_parameters(self, node: MCTSNode) -> Dict[str, float]:
        """Get current DPW parameters based on visit count.
        
        Args:
            node: Current node
            
        Returns:
            Dictionary of current parameters
        """
        for i, threshold in enumerate(self.visit_thresholds):
            if node.visits < threshold:
                return self.parameter_sets[i]
        return self.parameter_sets[-1]
        
    def _adjust_parameters(self, reward_history: List[float]) -> None:
        """Adjust policy parameters based on recent performance.
        
        Args:
            reward_history: List of recent rewards
        """
        if len(reward_history) < 5:
            return
            
        recent_rewards = reward_history[-5:]
        avg_reward = np.mean(recent_rewards)
        reward_trend = np.mean(np.diff(recent_rewards))
        
        if reward_trend < 0:
            self.exploration_constant = min(5.0, self.exploration_constant * 1.2)
        else:
            self.exploration_constant = max(1.0, self.exploration_constant * 0.9)
            
        if avg_reward < 100:
            self.k_action = min(20.0, self.k_action * 1.5)
            self.alpha_action = min(0.9, self.alpha_action + 0.1)
        else:
            self.k_action = max(5.0, self.k_action * 0.8)
            self.alpha_action = max(0.5, self.alpha_action - 0.05)
            
    def _evaluate_state(self, state: np.ndarray) -> float:
        """Enhanced state evaluation with multiple reward components.
        
        Args:
            state: Current state vector
            
        Returns:
            Combined reward value
        """
        width, height = self.grid_size
        grid_size = width * height
        
        belief_map = state[2:2+grid_size].reshape(height, width)
        uncertainty_map = state[2+grid_size:2+2*grid_size].reshape(height, width)
        visited_mask = state[2+2*grid_size:].reshape(height, width)
        pos_x, pos_y = state[:2]
        
        # 1. Enhanced Mutual Information Reward
        unvisited_mask = 1 - visited_mask
        mutual_info = np.sum(uncertainty_map * unvisited_mask)
        
        # Add bonus for high uncertainty regions
        high_uncertainty = np.where(uncertainty_map > np.percentile(uncertainty_map, 90))
        if len(high_uncertainty[0]) > 0:
            mutual_info *= 2.5  # Increased from 2.0
        
        # 2. Enhanced Novelty Reward
        novelty = np.sum(unvisited_mask) / grid_size
        # Add bonus for exploring new regions
        if novelty > 0.7:  # Early exploration phase
            novelty *= 4.0  # Increased from 3.0
        elif novelty < 0.3:  # Late exploration phase
            novelty *= 2.5  # Increased from 2.0
        
        # 3. Enhanced Frontier Reward
        frontier = self._calculate_frontier_reward(visited_mask)
        # Add bonus for expanding frontiers
        if frontier > 0:
            frontier *= 2.0  # Increased from 1.5
        
        # 4. Strategic Position Reward
        # Reward positions that are equidistant from explored regions
        dist_to_center = np.sqrt((pos_x - width/2)**2 + (pos_y - height/2)**2)
        position_reward = 0.0
        
        # Calculate distance to nearest visited cell
        visited_positions = np.where(visited_mask == 1)
        if len(visited_positions[0]) > 0:
            distances = np.sqrt(
                (visited_positions[1] - pos_x)**2 + 
                (visited_positions[0] - pos_y)**2
            )
            min_dist = np.min(distances)
            position_reward = np.exp(-min_dist/2.0)  # Changed from 3.0 to 2.0 for even faster decay
            
        # 5. Exploration Progress Reward
        progress = np.sum(visited_mask) / grid_size
        progress_reward = 0.0
        if progress > 0.5:  # Late exploration phase
            progress_reward = progress * 300.0  # Increased from 200.0
        
        # 6. Belief Quality Reward
        belief_quality = np.mean(belief_map)  # Average belief value
        belief_reward = belief_quality * 150.0  # Increased from 100.0
        
        # 7. Information Gain Reward
        info_gain = np.sum(uncertainty_map * (1 - visited_mask)) / np.sum(1 - visited_mask)
        info_gain_reward = info_gain * 200.0  # Increased from 150.0
        
        # 8. Exploration Efficiency Reward
        if len(visited_positions[0]) > 0:
            avg_dist = np.mean(distances)
            efficiency_reward = np.exp(-avg_dist/8.0) * 150.0  # Changed from 10.0 to 8.0 and increased from 100.0
        else:
            efficiency_reward = 0.0
        
        # Combine rewards with dynamic weights
        weights = {
            'mutual_info': 0.3,  # Increased from 0.25
            'novelty': 0.25,     # Increased from 0.2
            'frontier': 0.15,
            'position': 0.1,
            'progress': 0.1,
            'belief': 0.05,      # Decreased from 0.1
            'info_gain': 0.03,   # Decreased from 0.05
            'efficiency': 0.02   # Decreased from 0.05
        }
        
        # Adjust weights based on exploration progress
        if progress < 0.3:  # Early exploration
            weights['novelty'] *= 2.5  # Increased from 2.0
            weights['frontier'] *= 2.5  # Increased from 2.0
            weights['info_gain'] *= 2.0  # Increased from 1.5
        elif progress > 0.7:  # Late exploration
            weights['mutual_info'] *= 2.5  # Increased from 2.0
            weights['belief'] *= 2.5  # Increased from 2.0
            weights['progress'] *= 2.0  # Increased from 1.5
        
        # Calculate final reward
        reward = (
            weights['mutual_info'] * mutual_info +
            weights['novelty'] * novelty * 300.0 +  # Increased from 200.0
            weights['frontier'] * frontier * 30.0 +  # Increased from 20.0
            weights['position'] * position_reward * 150.0 +  # Increased from 100.0
            weights['progress'] * progress_reward +
            weights['belief'] * belief_reward +
            weights['info_gain'] * info_gain_reward +
            weights['efficiency'] * efficiency_reward
        )
        
        # Add small penalty for revisiting cells
        if visited_mask[int(pos_y), int(pos_x)] == 1:
            reward *= 0.6  # Increased penalty from 0.7
            
        return reward
        
    def _calculate_frontier_reward(self, visited_mask: np.ndarray) -> float:
        """Calculate frontier reward using 8-connected neighborhood.
        
        Args:
            visited_mask: Binary mask of visited cells
            
        Returns:
            Frontier reward value
        """
        height, width = visited_mask.shape
        frontier_reward = 0.0
        
        # 8-connected neighborhood offsets
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(height):
            for j in range(width):
                if visited_mask[i,j] == 1:
                    # Check neighbors
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < height and 0 <= nj < width and 
                            visited_mask[ni,nj] == 0):
                            frontier_reward += 1.0
                            
        return frontier_reward

    def _rollout(self, state: np.ndarray, depth: int) -> float:
        """Intelligent rollout using belief information.
        
        Args:
            state: Current state vector
            depth: Current depth
            
        Returns:
            Accumulated reward
        """
        if depth >= self.max_depth:
            return 0.0
            
        width, height = self.grid_size
        grid_size = width * height
        belief_map = state[2:2+grid_size].reshape(height, width)
        uncertainty_map = state[2+grid_size:2+2*grid_size].reshape(height, width)
        visited_mask = state[2+2*grid_size:].reshape(height, width)
        
        pos_x, pos_y = state[:2]
        
        # Find promising direction based on uncertainty
        unvisited_positions = np.where(visited_mask == 0)
        if len(unvisited_positions[0]) > 0:
            distances = np.sqrt(
                (unvisited_positions[1] - pos_x)**2 + 
                (unvisited_positions[0] - pos_y)**2
            )
            
            uncertainties = uncertainty_map[unvisited_positions]
            scores = uncertainties / (distances + 1)
            
            best_idx = np.argmax(scores)
            target_y, target_x = unvisited_positions[0][best_idx], unvisited_positions[1][best_idx]
            
            dx = np.clip(target_x - pos_x, -1, 1)
            dy = np.clip(target_y - pos_y, -1, 1)
            action = (dx, dy)
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            action = (np.cos(angle), np.sin(angle))
            
        next_state = self._get_next_state(state, action)
        reward = self._evaluate_state(next_state)
        
        return reward + self.discount_factor * self._rollout(next_state, depth + 1)

    def _get_action_type(self, dx: float, dy: float) -> RoverActionType:
        """Convert action vector to RoverActionType.
        
        Args:
            dx: x component of action
            dy: y component of action
            
        Returns:
            RoverActionType
        """
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