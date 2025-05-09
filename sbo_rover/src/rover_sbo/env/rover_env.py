"""Rover environment implementation."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from loguru import logger

from ..core.states import (
    RoverWorldState, RoverBelief, RoverLocationBelief,
    RoverPos
)
from ..core.actions import RoverAction, RoverActionType

def create_rover_env(
    grid_size: Tuple[int, int],
    num_sample_types: int,
    seed: Optional[int] = None
) -> 'RoverEnv':
    """Create a RoverEnv instance with simplified parameters.
    
    Args:
        grid_size: Size of the grid (width, height)
        num_sample_types: Number of different sample types
        seed: Random seed
        
    Returns:
        RoverEnv instance
    """
    rng = np.random.RandomState(seed)
    
    # Generate random location states
    n_locations = grid_size[0] * grid_size[1]
    location_states = rng.randint(0, num_sample_types, size=n_locations)
    
    # Generate random metadata
    location_metadata = [
        rng.normal(0, 1, size=num_sample_types)
        for _ in range(n_locations)
    ]
    
    # Set default parameters (scaled down by 100)
    init_pos = (0, 0)
    goal_pos = (grid_size[0]-1, grid_size[1]-1)
    
    return RoverEnv(
        map_size=grid_size,
        location_states=location_states,
        location_metadata=location_metadata,
        good_sample_reward=1.0,
        bad_sample_penalty=0.5,
        init_pos=init_pos,
        cost_budget=10.0,
        drill_time=0.1,
        step_size=1,
        discount_factor=0.95,
        sigma_drill=0.1,
        sigma_spec=0.2,
        goal_pos=goal_pos,
        rng=rng
    )

class RoverEnv:
    """Rover environment for autonomous exploration."""
    
    def __init__(
        self,
        map_size: Tuple[int, int],
        location_states: np.ndarray,
        location_metadata: List[np.ndarray],
        good_sample_reward: float,
        bad_sample_penalty: float,
        init_pos: Tuple[int, int],
        cost_budget: float,
        drill_time: float,
        step_size: int,
        discount_factor: float,
        sigma_drill: float,
        sigma_spec: float,
        goal_pos: Tuple[int, int],
        rng: np.random.RandomState
    ):
        """Initialize rover environment.
        
        Args:
            map_size: Size of the environment map
            location_states: States of each location
            location_metadata: Metadata for each location
            good_sample_reward: Reward for good samples
            bad_sample_penalty: Penalty for bad samples
            init_pos: Initial position of the rover
            cost_budget: Total cost budget
            drill_time: Time to drill a sample
            step_size: Size of each step
            discount_factor: Discount factor for rewards
            sigma_drill: Standard deviation for drill noise
            sigma_spec: Standard deviation for spectrometer noise
            goal_pos: Goal position
            rng: Random number generator
        """
        self.map_size = map_size
        self.location_states = location_states
        self.location_metadata = location_metadata
        self.good_sample_reward = good_sample_reward
        self.bad_sample_penalty = bad_sample_penalty
        self.init_pos = RoverPos(*init_pos)
        self.cost_budget = cost_budget
        self.drill_time = drill_time
        self.step_size = step_size
        self.discount_factor = discount_factor
        self.sigma_drill = sigma_drill
        self.sigma_spec = sigma_spec
        self.goal_pos = RoverPos(*goal_pos)
        self.rng = rng
        
        # Initialize state
        self.state = None
        self.belief = None
        
        # Precompute shortest paths
        self.shortest_paths = self._compute_shortest_paths()
        
    @property
    def grid_size(self) -> Tuple[int, int]:
        """Get the grid size.
        
        Returns:
            Grid size (width, height)
        """
        return self.map_size
        
    @property
    def state_dim(self) -> int:
        """Return the dimension of the state vector for the environment."""
        return len(self._get_state_vector())
        
    def reset(self) -> np.ndarray:
        """Reset the environment.
        
        Returns:
            Initial state vector
        """
        # Initialize world state
        self.state = RoverWorldState(
            pos=self.init_pos,
            visited={self.init_pos},
            location_states=self.location_states,
            cost_expended=0.0,
            drill_samples=[]
        )
        
        # Initialize belief state
        self.belief = self.initial_belief_state()
        
        # Convert to state vector
        return self._get_state_vector()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action: Action vector (dx, dy)
            
        Returns:
            Tuple of:
                - Next state vector
                - Reward
                - Done flag
                - Info dictionary
        """
        # Convert action vector to RoverAction
        dx, dy = action
        current_pos = self.state.pos
        target_pos = RoverPos(
            x=int(round(current_pos.x + dx)),
            y=int(round(current_pos.y + dy))
        )
        
        # Clip target position to map bounds
        target_pos = RoverPos(
            x=np.clip(target_pos.x, 0, self.map_size[0] - 1),
            y=np.clip(target_pos.y, 0, self.map_size[1] - 1)
        )
        
        # Create action
        if target_pos.x > current_pos.x:
            action_type = RoverActionType.RIGHT
        elif target_pos.x < current_pos.x:
            action_type = RoverActionType.LEFT
        elif target_pos.y > current_pos.y:
            action_type = RoverActionType.UP
        elif target_pos.y < current_pos.y:
            action_type = RoverActionType.DOWN
        else:
            action_type = RoverActionType.WAIT
            
        rover_action = RoverAction(
            action_type=action_type,
            target_pos=(target_pos.x, target_pos.y)
        )
        
        # Generate next state
        next_state = self.generate_s(self.state, rover_action)
        
        # Calculate reward
        reward = self._calculate_reward(self.state, next_state)
        
        # Update belief
        next_belief = self._update_belief(self.belief, rover_action, next_state)
        
        # Update current state and belief
        self.state = next_state
        self.belief = next_belief
        
        # Check if done
        done = (next_state.cost_expended >= self.cost_budget or
                next_state.pos == self.goal_pos)
                
        # Create info dictionary
        info = {
            "belief": next_belief,
            "cost_expended": next_state.cost_expended,
            "visited_locations": len(next_state.visited)
        }
        
        return self._get_state_vector(), reward, done, info
        
    def render(self, ax=None):
        """Render the environment.
        
        Args:
            ax: Matplotlib axis to render on
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
            
        # Plot location states
        states_grid = self.location_states.reshape(self.map_size)
        ax.imshow(states_grid, cmap='viridis')
        
        # Plot visited locations
        if self.state is not None:
            visited_x = [pos.x for pos in self.state.visited]
            visited_y = [pos.y for pos in self.state.visited]
            ax.plot(visited_x, visited_y, 'r.', markersize=10, alpha=0.5)
            
        # Plot current position
        if self.state is not None:
            ax.plot(self.state.pos.x, self.state.pos.y, 'ro', markersize=15)
            
        # Plot goal position
        ax.plot(self.goal_pos.x, self.goal_pos.y, 'g*', markersize=15)
        
        ax.grid(True)
        ax.set_xticks(range(self.map_size[0]))
        ax.set_yticks(range(self.map_size[1]))
        
    def _get_state_vector(self) -> np.ndarray:
        """Convert current state to vector representation.
        
        Returns:
            State vector with format:
            [pos_x, pos_y, location_states, uncertainty_map, visited_mask]
        """
        if self.state is None:
            width, height = self.map_size
            grid_elements = width * height
            return np.zeros(2 + 3 * grid_elements)
            
        # Current position
        state_vec = np.array([self.state.pos.x, self.state.pos.y])
        
        # Flatten location states
        state_vec = np.concatenate([state_vec, self.location_states.flatten()])
        
        # Uncertainty map (initially uniform)
        width, height = self.map_size
        grid_elements = width * height
        uncertainty_map = np.ones(grid_elements)  # Maximum uncertainty initially
        state_vec = np.concatenate([state_vec, uncertainty_map])
        
        # Visited mask
        visited_mask = np.zeros(self.map_size)
        for pos in self.state.visited:
            visited_mask[pos.y, pos.x] = 1
        state_vec = np.concatenate([state_vec, visited_mask.flatten()])
        
        return state_vec
        
    def _calculate_reward(
        self,
        state: RoverWorldState,
        next_state: RoverWorldState
    ) -> float:
        """Calculate reward for a transition.
        
        Args:
            state: Current state
            next_state: Next state
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Calculate exploration progress
        total_cells = self.map_size[0] * self.map_size[1]
        progress = len(next_state.visited) / total_cells
        
        # Enhanced reward components
        # 1. Mutual Information Reward
        mi_reward = 0.0
        if next_state.pos not in state.visited:
            # Calculate uncertainty reduction
            current_uncertainty = self._calculate_uncertainty(state)
            next_uncertainty = self._calculate_uncertainty(next_state)
            mi_reward = 0.025 * (current_uncertainty - next_uncertainty)
        
        # 2. Novelty Reward
        novelty_reward = 0.0
        if next_state.pos not in state.visited:
            if progress < 0.3:  # Early exploration
                novelty_reward = 4.0 * 3.0
            else:
                novelty_reward = 3.0
        
        # 3. Frontier Reward
        frontier_reward = 0.0
        if self._is_frontier_cell(next_state.pos, next_state.visited):
            frontier_reward = 2.0 * 0.3
        
        # 4. Strategic Position Reward
        pos_reward = 0.0
        if next_state.pos not in state.visited:
            # Calculate distance to nearest explored region
            min_dist = self._get_min_distance_to_explored(next_state.pos, state.visited)
            pos_reward = 1.5 * np.exp(-2.0 * min_dist)
        
        # 5. Exploration Progress Reward
        progress_reward = 0.0
        if progress > 0.7:  # Late exploration
            progress_reward = 2.0 * 3.0
        elif progress > 0.3:  # Mid exploration
            progress_reward = 2.0
        else:  # Early exploration
            progress_reward = 1.0
        
        # 6. Belief Quality Reward
        belief_reward = 0.0
        if next_state.pos not in state.visited:
            belief_reward = 1.5 * self._calculate_belief_quality(next_state)
        
        # 7. Information Gain Reward
        info_reward = 0.0
        if next_state.pos not in state.visited:
            info_reward = 2.0 * self._calculate_information_gain(next_state)
        
        # 8. Exploration Efficiency Reward
        efficiency_reward = 0.0
        if next_state.pos not in state.visited:
            avg_dist = self._calculate_average_distance(next_state.pos, state.visited)
            efficiency_reward = 1.0 * np.exp(-8.0 * avg_dist)
        
        # Dynamic weights based on exploration progress
        weights = {
            'mi': 0.3,
            'novelty': 0.25,
            'frontier': 0.15,
            'position': 0.1,
            'progress': 0.05,
            'belief': 0.05,
            'info': 0.03,
            'efficiency': 0.02
        }
        
        # Apply weights and multipliers
        if progress < 0.3:  # Early exploration
            weights['novelty'] *= 2.5
            weights['frontier'] *= 2.5
            weights['position'] *= 2.5
        elif progress > 0.7:  # Late exploration
            weights['progress'] *= 2.5
            weights['belief'] *= 2.5
            weights['info'] *= 2.5
        
        # Calculate final reward
        reward = (
            weights['mi'] * mi_reward +
            weights['novelty'] * novelty_reward +
            weights['frontier'] * frontier_reward +
            weights['position'] * pos_reward +
            weights['progress'] * progress_reward +
            weights['belief'] * belief_reward +
            weights['info'] * info_reward +
            weights['efficiency'] * efficiency_reward
        )
        
        # Penalty for revisiting cells
        if next_state.pos in state.visited:
            reward *= 0.6
        
        return reward
        
    def _calculate_uncertainty(self, state: RoverWorldState) -> float:
        """Calculate uncertainty of the current state.
        
        Args:
            state: Current state
            
        Returns:
            Uncertainty value
        """
        # Calculate entropy of location beliefs
        entropy = 0.0
        for pos in state.visited:
            idx = self._pos_to_idx(pos)
            probs = self.belief.location_beliefs[idx].probs
            entropy -= np.sum(probs * np.log2(probs + 1e-10))
        return entropy
        
    def _is_frontier_cell(self, pos: RoverPos, visited: Set[RoverPos]) -> bool:
        """Check if a cell is a frontier cell.
        
        Args:
            pos: Position to check
            visited: Set of visited positions
            
        Returns:
            True if the cell is a frontier cell
        """
        # Check 8-connected neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                x, y = pos.x + dx, pos.y + dy
                if (0 <= x < self.map_size[0] and 
                    0 <= y < self.map_size[1] and 
                    RoverPos(x, y) not in visited):
                    return True
        return False
        
    def _get_min_distance_to_explored(self, pos: RoverPos, visited: Set[RoverPos]) -> float:
        """Calculate minimum distance to explored regions.
        
        Args:
            pos: Current position
            visited: Set of visited positions
            
        Returns:
            Minimum distance
        """
        if not visited:
            return 0.0
        min_dist = float('inf')
        for v_pos in visited:
            dist = np.sqrt((pos.x - v_pos.x)**2 + (pos.y - v_pos.y)**2)
            min_dist = min(min_dist, dist)
        return min_dist
        
    def _calculate_belief_quality(self, state: RoverWorldState) -> float:
        """Calculate quality of current belief.
        
        Args:
            state: Current state
            
        Returns:
            Belief quality value
        """
        # Calculate average confidence in beliefs
        confidence = 0.0
        for pos in state.visited:
            idx = self._pos_to_idx(pos)
            probs = self.belief.location_beliefs[idx].probs
            confidence += np.max(probs)
        return confidence / len(state.visited) if state.visited else 0.0
        
    def _calculate_information_gain(self, state: RoverWorldState) -> float:
        """Calculate information gain from current state.
        
        Args:
            state: Current state
            
        Returns:
            Information gain value
        """
        # Calculate mutual information between visited and unvisited regions
        mi = 0.0
        for pos in state.visited:
            idx = self._pos_to_idx(pos)
            probs = self.belief.location_beliefs[idx].probs
            mi += np.sum(probs * np.log2(probs + 1e-10))
        return -mi
        
    def _calculate_average_distance(self, pos: RoverPos, visited: Set[RoverPos]) -> float:
        """Calculate average distance to visited regions.
        
        Args:
            pos: Current position
            visited: Set of visited positions
            
        Returns:
            Average distance
        """
        if not visited:
            return 0.0
        total_dist = 0.0
        for v_pos in visited:
            dist = np.sqrt((pos.x - v_pos.x)**2 + (pos.y - v_pos.y)**2)
            total_dist += dist
        return total_dist / len(visited)
        
    def _update_belief(
        self,
        belief: RoverBelief,
        action: RoverAction,
        next_state: RoverWorldState
    ) -> RoverBelief:
        """Update belief state.
        
        Args:
            belief: Current belief
            action: Action taken
            next_state: Next world state
            
        Returns:
            Updated belief
        """
        # Update visited locations
        new_visited = belief.visited.copy()
        new_visited.add(next_state.pos)
        
        # Update location beliefs
        new_location_beliefs = self.belief_update_location_states_visit(
            belief.location_beliefs,
            next_state.pos
        )
        
        # Update drill samples
        new_drill_samples = belief.drill_samples.copy()
        if action.action_type == RoverActionType.DRILL:
            new_drill_samples.append((
                RoverPos(*action.target_pos),
                next_state.drill_samples[-1][1]
            ))
            
        return RoverBelief(
            pos=next_state.pos,
            visited=new_visited,
            location_beliefs=new_location_beliefs,
            cost_expended=next_state.cost_expended,
            drill_samples=new_drill_samples
        )
        
    def _compute_shortest_paths(self) -> np.ndarray:
        """Compute shortest paths between all locations.
        
        Returns:
            Matrix of shortest path distances
        """
        n_locations = self.map_size[0] * self.map_size[1]
        dist = np.full((n_locations, n_locations), np.inf)
        
        # Initialize distances
        for i in range(n_locations):
            dist[i, i] = 0
            pos_i = self._idx_to_pos(i)
            
            # Add edges to adjacent locations
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                x, y = pos_i.x + dx, pos_i.y + dy
                if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                    j = self._pos_to_idx(RoverPos(x, y))
                    dist[i, j] = 1
                    
        # Floyd-Warshall algorithm
        for k in range(n_locations):
            for i in range(n_locations):
                for j in range(n_locations):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        
        return dist
        
    def _pos_to_idx(self, pos: RoverPos) -> int:
        """Convert position to index.
        
        Args:
            pos: Position
            
        Returns:
            Index
        """
        return pos.y * self.map_size[0] + pos.x
        
    def _idx_to_pos(self, idx: int) -> RoverPos:
        """Convert index to position.
        
        Args:
            idx: Index
            
        Returns:
            Position
        """
        x = idx % self.map_size[0]
        y = idx // self.map_size[0]
        return RoverPos(x, y)
        
    def initial_belief_state(self) -> RoverBelief:
        """Get initial belief state.
        
        Returns:
            Initial belief state
        """
        # Initialize location beliefs with uniform distribution
        n_locations = self.map_size[0] * self.map_size[1]
        location_beliefs = [
            RoverLocationBelief(probs=np.ones(4) / 4)
            for _ in range(n_locations)
        ]
        
        return RoverBelief(
            pos=self.init_pos,
            visited={self.init_pos},
            location_beliefs=location_beliefs,
            cost_expended=0.0,
            drill_samples=[]
        )
        
    def generate_s(self, state: RoverWorldState, action: RoverAction) -> RoverWorldState:
        """Generate next state after taking action.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state
        """
        try:
            # Get next position
            new_pos = self._get_next_pos(state.pos, action)
            
            # Update visited set
            new_visited = state.visited.copy()
            new_visited.add(new_pos)
            
            # Update cost
            new_cost = state.cost_expended
            if action.action_type == RoverActionType.DRILL:
                new_cost += self.drill_time
            else:
                new_cost += 1.0  # Movement cost
                
            # Update drill samples
            new_drill_samples = state.drill_samples.copy()
            if action.action_type == RoverActionType.DRILL:
                target_pos = action.target_pos  # Already a RoverPos object
                idx = self._pos_to_idx(target_pos)
                sample_value = self.location_states[idx]
                new_drill_samples.append((target_pos, sample_value))
                
            return RoverWorldState(
                pos=new_pos,
                visited=new_visited,
                location_states=self.location_states,
                cost_expended=new_cost,
                drill_samples=new_drill_samples
            )
            
        except Exception as e:
            logger.error(f"Error in generate_s: {str(e)}")
            return state
            
    def _get_next_pos(self, pos: RoverPos, action: RoverAction) -> RoverPos:
        """Get next position after taking an action.
        
        Args:
            pos: Current position
            action: Action to take
            
        Returns:
            Next position
        """
        if action.action_type == RoverActionType.UP:
            return RoverPos(pos.x, min(pos.y + 1, self.map_size[1] - 1))
        elif action.action_type == RoverActionType.DOWN:
            return RoverPos(pos.x, max(pos.y - 1, 0))
        elif action.action_type == RoverActionType.LEFT:
            return RoverPos(max(pos.x - 1, 0), pos.y)
        elif action.action_type == RoverActionType.RIGHT:
            return RoverPos(min(pos.x + 1, self.map_size[0] - 1), pos.y)
        return pos
        
    def belief_update_location_states_visit(
        self,
        location_beliefs: List[RoverLocationBelief],
        pos: RoverPos
    ) -> List[RoverLocationBelief]:
        """Update location beliefs after visiting a location.
        
        Args:
            location_beliefs: Current location beliefs
            pos: Visited position
            
        Returns:
            Updated location beliefs
        """
        try:
            new_beliefs = location_beliefs.copy()
            idx = self._pos_to_idx(pos)
            
            # Update belief for visited location
            state = self.location_states[idx]
            probs = np.zeros(4)
            if state >= 0.8:  # Good sample
                probs[0] = 1.0
            elif state <= 0.1:  # Bad sample
                probs[1] = 1.0
            elif 0.4 <= state <= 0.6:  # Beacon
                probs[2] = 1.0
            else:  # Empty
                probs[3] = 1.0
                
            new_beliefs[idx] = RoverLocationBelief(probs=probs)
            
            return new_beliefs
            
        except Exception as e:
            logger.error(f"Error in belief_update_location_states_visit: {str(e)}")
            return location_beliefs 