from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
import numpy as np
from scipy.stats import entropy

from .actions import MultimodalIPPAction, ISRSSensor
from .states import ISRSWorldState, ISRSObservation, ISRSBelief, ISRS_STATE
from .env import ISRSEnv

@dataclass
class RewardComponents:
    """Components of the enhanced reward structure."""
    base_reward: float
    info_gain: float
    exploration_bonus: float
    progress_factor: float
    distance_penalty: float
    belief_bonus: float

@dataclass
class ISRSPOMDP:
    """POMDP wrapper for ISRS environment."""
    
    def __init__(
        self,
        env: ISRSEnv,
        discount_factor: float = 0.95,
        info_gain_weight: float = 0.3,
        distance_penalty_weight: float = 0.2,
        exploration_bonus: float = 5.0,
        progressive_factor: float = 1.2,
        belief_weight: float = 0.2,
        uncertainty_threshold: float = 0.1,
        total_budget: float = 100.0,
        max_steps: int = 100,
        max_cost: float = 100.0,
        min_uncertainty: float = 0.1
    ) -> None:
        """Initialize POMDP.
        
        Args:
            env: Environment instance
            discount_factor: Discount factor for future rewards
            info_gain_weight: Weight for information gain reward
            distance_penalty_weight: Weight for distance penalty
            exploration_bonus: Bonus for exploring new locations
            progressive_factor: Factor for progressive reward
            belief_weight: Weight for belief-based reward
            uncertainty_threshold: Threshold for uncertainty
            total_budget: Total budget for actions
            max_steps: Maximum number of steps before termination
            max_cost: Maximum cost before termination
            min_uncertainty: Minimum uncertainty before termination
        """
        self.env = env
        self.discount_factor = discount_factor
        self.info_gain_weight = info_gain_weight
        self.distance_penalty_weight = distance_penalty_weight
        self.exploration_bonus = exploration_bonus
        self.progressive_factor = progressive_factor
        self.belief_weight = belief_weight
        self.uncertainty_threshold = uncertainty_threshold
        self.total_budget = total_budget
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.min_uncertainty = min_uncertainty
        
    def reset(self) -> Tuple[ISRSWorldState, ISRSObservation, ISRSBelief]:
        """Reset POMDP.
        
        Returns:
            Tuple of (initial state, initial observation, initial belief)
        """
        return self.env.reset()
        
    def step(
        self,
        action: MultimodalIPPAction
    ) -> Tuple[ISRSWorldState, ISRSObservation, float, bool, ISRSBelief]:
        """Take a step in the POMDP.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next state, observation, reward, done, belief)
        """
        return self.env.step(action)
        
    def get_actions(self, state: ISRSWorldState) -> List[MultimodalIPPAction]:
        """Get available actions for current state.
        
        Args:
            state: Current world state
            
        Returns:
            List of available actions
        """
        actions = []
        
        # Add visit actions for unvisited locations
        for i in range(len(state.location_states)):
            if i not in state.visited:
                # Calculate cost based on distance
                cost = self.env.shortest_paths[state.current, i]
                actions.append(MultimodalIPPAction(
                    visit_location=i,
                    sensing_action=None,
                    cost=cost
                ))
        
        # Add sensing actions for current location
        actions.extend([
            MultimodalIPPAction(
                visit_location=None,
                sensing_action=ISRSSensor.LOW,
                cost=0.1
            ),
            MultimodalIPPAction(
                visit_location=None,
                sensing_action=ISRSSensor.MEDIUM,
                cost=0.2
            ),
            MultimodalIPPAction(
                visit_location=None,
                sensing_action=ISRSSensor.HIGH,
                cost=0.3
            )
        ])
        
        return actions
        
    def is_terminal(self, state: ISRSWorldState) -> bool:
        """Check if state is terminal.
        
        Args:
            state: Current state
            
        Returns:
            True if state is terminal
        """
        # Check if max steps reached
        if len(state.visited) >= self.max_steps:
            return True
        
        # Check if all good rocks visited
        if all(i in state.visited for i, val in enumerate(state.location_states) if val == ISRS_STATE.RSGOOD):
            return True
        
        # Check if cost exceeded
        if state.cost_expended >= self.max_cost:
            return True
        
        # Check if belief uncertainty is low enough
        if self._get_belief_uncertainty(state) < self.min_uncertainty:
            return True
        
        return False

    def _get_belief_uncertainty(self, state: ISRSWorldState) -> float:
        """Calculate belief uncertainty.
        
        Args:
            state: Current state
            
        Returns:
            Belief uncertainty measure
        """
        # Calculate entropy of location states
        entropy = 0.0
        for i, loc_state in enumerate(state.location_states):
            if i not in state.visited:
                # Calculate entropy for unvisited locations
                if loc_state == ISRS_STATE.RSGOOD:
                    p_good = 1.0
                    p_bad = 0.0
                elif loc_state == ISRS_STATE.RSBAD:
                    p_good = 0.0
                    p_bad = 1.0
                else:
                    # For unknown or other states, assume uniform distribution
                    p_good = 0.5
                    p_bad = 0.5
                
                if p_good > 0:
                    entropy -= p_good * np.log2(p_good)
                if p_bad > 0:
                    entropy -= p_bad * np.log2(p_bad)
        
        return entropy
        
    def get_reward(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction,
        next_state: ISRSWorldState
    ) -> float:
        """Calculate reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Base reward from environment
        reward += self.env._calculate_reward(next_state, action)
        
        # Information gain reward
        if action.sensing_action is not None:
            info_gain = self._calculate_info_gain(state, action, next_state)
            reward += self.info_gain_weight * info_gain
            
        # Distance penalty
        if action.visit_location is not None:
            distance = self._calculate_distance(state.current, action.visit_location)
            reward -= self.distance_penalty_weight * distance
            
        # Exploration bonus
        if action.visit_location is not None and action.visit_location not in state.visited:
            reward += self.exploration_bonus
            
        # Progressive reward
        progress = len(next_state.visited) / self.env.n_locations
        reward *= self.progressive_factor ** progress
        
        return reward
        
    def _calculate_info_gain(
        self,
        s: ISRSWorldState,
        a: MultimodalIPPAction,
        sp: ISRSWorldState
    ) -> float:
        """Calculate information gain from action.
        
        Args:
            s: Current state
            a: Action taken
            sp: Next state
            
        Returns:
            Information gain value
        """
        if a.sensing_action is None:
            return 0.0
            
        # Calculate entropy reduction
        current_entropy = self._calculate_state_entropy(s)
        next_entropy = self._calculate_state_entropy(sp)
        info_gain = max(0, current_entropy - next_entropy)
        return info_gain * 2.0  # Double the information gain reward
        
    def _calculate_entropy(self, state: ISRSWorldState) -> float:
        """Calculate entropy of state.
        
        Args:
            state: Current state
            
        Returns:
            Entropy value
        """
        # Count state frequencies
        counts = {s: 0 for s in ISRS_STATE}
        total = 0
        for s in state.location_states:
            if s != ISRS_STATE.UNKNOWN:
                counts[s] += 1
                total += 1
                
        # Calculate entropy
        entropy = 0.0
        if total > 0:
            for count in counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log(p)
                    
        return entropy
        
    def _calculate_distance(self, from_loc: int, to_loc: int) -> float:
        """Calculate distance between locations.
        
        Args:
            from_loc: Starting location index
            to_loc: Target location index
            
        Returns:
            Distance value
        """
        from_coord = self.env.location_metadata[from_loc]
        to_coord = self.env.location_metadata[to_loc]
        return np.sqrt(
            (from_coord[0] - to_coord[0]) ** 2 +
            (from_coord[1] - to_coord[1]) ** 2
        )

    def generate_s(self, s: Union[ISRSWorldState, ISRSBelief], a: MultimodalIPPAction, rng: np.random.RandomState) -> ISRSWorldState:
        """Generate next state given current state and action.
        
        Args:
            s: Current state or belief state
            a: Action taken
            rng: Random number generator
            
        Returns:
            Next world state
        """
        # If input is a belief state, sample a world state
        if isinstance(s, ISRSBelief):
            s = s.sample_state(rng)
        
        new_visited = s.visited.copy()
        new_cost = s.cost_expended + a.cost
        new_current = s.current

        if a.visit_location is not None:
            new_current = a.visit_location
            new_visited.add(a.visit_location)

        return ISRSWorldState(
            current=new_current,
            visited=new_visited,
            location_states=s.location_states.copy(),
            cost_expended=new_cost
        )

    def generate_o(self, s: ISRSWorldState, a: MultimodalIPPAction, sp: ISRSWorldState, rng: np.random.RandomState) -> ISRSObservation:
        """Generate observation given current state, action, and next state.
        
        Args:
            s: Current state
            a: Action taken
            sp: Next state
            rng: Random number generator
            
        Returns:
            Generated observation
        """
        if a.visit_location is not None:
            # Perfect observation of visited location
            obs_states = sp.location_states.copy()
            obs_weight = 1.0
        else:
            # Noisy observation from sensor
            obs_states = [ISRS_STATE.UNKNOWN] * len(sp.location_states)
            if rng.random() < a.sensing_action.value:
                obs_states[sp.current] = sp.location_states[sp.current]
            obs_weight = a.sensing_action.value
        
        # Update GP features
        self.env.update_gp_features(sp)
        
        return ISRSObservation(
            current=sp.current,
            visited=sp.visited,
            location_states=obs_states,
            cost_expended=sp.cost_expended,
            obs_weight=obs_weight
        )

    def get_reward(
        self,
        s: Union[ISRSWorldState, ISRSBelief],
        a: MultimodalIPPAction,
        sp: ISRSWorldState
    ) -> Tuple[float, RewardComponents]:
        """Calculate enhanced reward with components.
        
        Args:
            s: Current state or belief state
            a: Action taken
            sp: Next state
            
        Returns:
            Tuple of (total reward, reward components)
        """
        # If input is a belief state, sample a world state
        if isinstance(s, ISRSBelief):
            s = s.sample_state(self.rng)

        if sp.cost_expended > self.total_budget:
            return -1000.0, RewardComponents(
                base_reward=-1000.0,
                info_gain=0.0,
                exploration_bonus=0.0,
                progress_factor=0.0,
                distance_penalty=0.0,
                belief_bonus=0.0
            )

        # Base reward for finding good rocks
        base_reward = 0.0
        if a.visit_location is not None and sp.location_states[a.visit_location] == 1:  # RSGOOD
            base_reward = 10.0

        # Calculate information gain
        info_gain = self._calculate_info_gain(s, a, sp)

        # Calculate exploration bonus
        exploration_bonus = self._calculate_exploration_bonus(s, a)

        # Calculate progress factor
        progress_factor = self._calculate_progress_factor(s)

        # Calculate distance penalty
        distance_penalty = self._calculate_distance_penalty(s, a)

        # Calculate belief bonus
        belief_bonus = self._calculate_belief_bonus(s, a)

        # Get adaptive weights
        weights = self._get_adaptive_weights(s)

        # Calculate total reward
        total_reward = (
            base_reward +
            weights['info_gain'] * info_gain +
            weights['exploration'] * exploration_bonus +
            weights['progressive'] * progress_factor -
            weights['distance'] * distance_penalty +
            weights['belief'] * belief_bonus
        )

        return total_reward, RewardComponents(
            base_reward=base_reward,
            info_gain=info_gain,
            exploration_bonus=exploration_bonus,
            progress_factor=progress_factor,
            distance_penalty=distance_penalty,
            belief_bonus=belief_bonus
        )

    def _calculate_state_entropy(self, s: ISRSWorldState) -> float:
        """Calculate entropy of state."""
        probs = []
        for i, state in enumerate(s.location_states):
            if i in s.visited:
                probs.append(1.0 if state == 1 else 0.0)  # Known state
            else:
                probs.append(0.5)  # Unknown state
        return entropy(probs)

    def _calculate_exploration_bonus(self, s: ISRSWorldState, a: MultimodalIPPAction) -> float:
        """Calculate exploration bonus for visiting new locations."""
        if a.visit_location is None or a.visit_location in s.visited:
            return 0.0

        # Calculate bonus based on distance to nearest unvisited location
        min_dist = float('inf')
        for loc in range(len(s.location_states)):
            if loc not in s.visited:
                dist = self.env.shortest_paths[s.current, loc]
                min_dist = min(min_dist, dist)

        return self.exploration_bonus / (1 + min_dist)

    def _calculate_progress_factor(self, s: ISRSWorldState) -> float:
        """Calculate progress factor based on visited locations."""
        return len(s.visited) / len(s.location_states)

    def _calculate_distance_penalty(self, s: ISRSWorldState, a: MultimodalIPPAction) -> float:
        """Calculate distance penalty for movement."""
        if a.visit_location is None:
            return 0.0
        return self.env.shortest_paths[s.current, a.visit_location]

    def _calculate_belief_bonus(self, s: ISRSWorldState, a: MultimodalIPPAction) -> float:
        """Calculate bonus based on belief state."""
        if a.visit_location is None:
            return 0.0

        # Get GP features for target location
        if self.env.gp_features is None:
            return 0.0
            
        # Higher bonus for locations with high uncertainty
        return self.env.gp_features.uncertainty

    def _get_adaptive_weights(self, s: ISRSWorldState) -> Dict[str, float]:
        """Get adaptive weights based on progress."""
        progress = self._calculate_progress_factor(s)
        
        # Adjust weights based on progress
        return {
            'info_gain': self.info_gain_weight * (1 - progress),
            'exploration': self.exploration_bonus * (1 - progress),
            'progressive': self.progressive_factor * progress,
            'distance': self.distance_penalty_weight * (1 - progress),
            'belief': self.belief_weight * progress
        }

    def initial_belief_state(self) -> ISRSBelief:
        """Generate initial belief state."""
        return self.env.initial_belief_state()

    def update_belief(self, b: ISRSBelief, a: MultimodalIPPAction, o: ISRSObservation) -> ISRSBelief:
        """Update belief state with action and observation."""
        return b.update(a, o, self.env)

    def sample_initial_state(self, rng: np.random.RandomState) -> ISRSWorldState:
        """Sample initial state."""
        location_states = self.env.sample_location_states(rng)
        return ISRSWorldState(
            current=0,  # Start at origin
            visited=set(),
            location_states=location_states,
            cost_expended=0.0
        ) 