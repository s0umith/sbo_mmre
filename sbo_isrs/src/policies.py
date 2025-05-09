"""Enhanced POMCP policy implementations for ISRS environment."""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from loguru import logger

# Local imports
from .states import ISRSWorldState, ISRSObservation, ISRSBelief, ISRS_STATE
from .actions import MultimodalIPPAction
from .pomdp import ISRSPOMDP
from .belief import ISRSBelief
from .enhanced_components import ObservationWidener, BeliefStateCluster, GaussianProcessBelief
from .enhanced_observations import EnhancedObservation
from .parallel_simulator import ParallelSimulator, SimulationConfig, RolloutStrategy
from .reward_components import EnhancedRewardCalculator

@dataclass
class POMCPNode:
    """Node in POMCP search tree.
    
    Attributes:
        action: Action that led to this node
        parent: Parent node
        belief: Belief state at this node
        rng: Random number generator
        children: Dictionary of child nodes
        visits: Number of visits to this node
        value: Total value accumulated at this node
    """
    action: Optional[MultimodalIPPAction]
    parent: Optional['POMCPNode']
    belief: ISRSBelief
    rng: np.random.RandomState
    children: Dict[MultimodalIPPAction, 'POMCPNode'] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    
    def get_child(self, action: MultimodalIPPAction) -> Optional['POMCPNode']:
        """Get child node for action.
        
        Args:
            action: Action to get child for
            
        Returns:
            Child node if exists, None otherwise
        """
        return self.children.get(action)
        
    def add_child(self, action: MultimodalIPPAction, child: 'POMCPNode') -> None:
        """Add child node.
        
        Args:
            action: Action that leads to child
            child: Child node
        """
        self.children[action] = child

class BasePolicy:
    """Base class for all policies.
    
    Attributes:
        pomdp: POMDP model
        rng: Random number generator
        num_simulations: Number of simulations
        max_depth: Maximum search depth
        exploration_constant: UCB exploration constant
    """
    def __init__(
        self,
        pomdp: Any,
        rng: np.random.Generator,
        num_simulations: int = 1000,
        max_depth: int = 50,
        exploration_constant: float = 1.0,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize base policy.
        
        Args:
            pomdp: POMDP model
            rng: Random number generator
            num_simulations: Number of simulations
            max_depth: Maximum search depth
            exploration_constant: UCB exploration constant
            **kwargs: Additional arguments
        """
        self.pomdp = pomdp
        self.rng = rng
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant

class EnhancedPOMCPPolicy(BasePolicy):
    """Enhanced base class for POMCP policies with additional features.
    
    Attributes:
        pomdp: POMDP model
        rng: Random number generator
        num_simulations: Number of simulations
        max_depth: Maximum search depth
        exploration_constant: UCB exploration constant
        discount_factor: Discount factor for rewards
        progressive_factor: Base progressive factor
        alpha_action: Action widening exponent
        alpha_observation: Observation widening exponent
    """
    def __init__(
        self,
        pomdp: Any,
        rng: np.random.Generator,
        num_simulations: int = 1000,
        max_depth: int = 50,
        exploration_constant: float = 1.0,
        discount_factor: float = 0.95,
        progressive_factor: float = 0.5,
        alpha_action: float = 0.5,
        alpha_observation: float = 0.5,
        width: float = 0.1,
        similarity_threshold: float = 0.6,
        max_width: float = 0.5,
        min_width: float = 0.01,
        width_decay: float = 0.95,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize enhanced POMCP policy.
        
        Args:
            pomdp: POMDP model
            rng: Random number generator
            num_simulations: Number of simulations
            max_depth: Maximum search depth
            exploration_constant: UCB exploration constant
            discount_factor: Discount factor for rewards
            progressive_factor: Base progressive factor
            alpha_action: Action widening exponent
            alpha_observation: Observation widening exponent
            width: Initial width parameter
            similarity_threshold: Threshold for observation similarity
            max_width: Maximum width for widening
            min_width: Minimum width for widening
            width_decay: Decay rate for width
            **kwargs: Additional arguments
        """
        super().__init__(
            pomdp=pomdp,
            rng=rng,
            num_simulations=num_simulations,
            max_depth=max_depth,
            exploration_constant=exploration_constant
        )
        self.discount_factor = discount_factor
        self.progressive_factor = progressive_factor
        self.alpha_action = alpha_action
        self.alpha_observation = alpha_observation
        
        # Initialize observation widener
        self.observation_widener = ObservationWidener(
            width=width,
            similarity_threshold=similarity_threshold,
            max_width=max_width,
            min_width=min_width,
            width_decay=width_decay
        )
        
        # Initialize belief cluster
        self.belief_cluster = BeliefStateCluster(
            similarity_threshold=kwargs.get('belief_similarity_threshold', 0.7),
            max_size=kwargs.get('max_belief_size', 100)
        )
        
        # Initialize GP belief
        self.gp_belief = GaussianProcessBelief(
            noise_level=kwargs.get('gp_noise_level', 1e-4),
            n_restarts=kwargs.get('gp_n_restarts', 10)
        )
        
        # Initialize parallel simulator
        self.parallel_simulator = ParallelSimulator(
            pomdp=pomdp,
            rng=rng,
            config=SimulationConfig(
                max_depth=kwargs.get('max_depth', 5),
                discount_factor=kwargs.get('discount_factor', 0.95),
                num_processes=kwargs.get('num_processes', None),
                batch_size=kwargs.get('batch_size', 100),
                rollout_strategy=kwargs.get('rollout_strategy', RolloutStrategy.RANDOM),
                show_progress=kwargs.get('show_progress', True),
                exploration_constant=exploration_constant,
                progressive_factor=progressive_factor,
                alpha_action=alpha_action,
                alpha_observation=alpha_observation,
                width=width,
                similarity_threshold=similarity_threshold,
                max_width=max_width,
                min_width=min_width,
                width_decay=width_decay,
                belief_similarity_threshold=kwargs.get('belief_similarity_threshold', 0.7),
                max_belief_size=kwargs.get('max_belief_size', 100),
                gp_noise_level=kwargs.get('gp_noise_level', 1e-4),
                gp_n_restarts=kwargs.get('gp_n_restarts', 10),
                info_gain_weight=kwargs.get('info_gain_weight', 0.3),
                exploration_weight=kwargs.get('exploration_weight', 0.5),
                progressive_weight=kwargs.get('progressive_weight', 0.3),
                distance_penalty_weight=kwargs.get('distance_penalty_weight', 0.2),
                belief_weight=kwargs.get('belief_weight', 0.4),
                exploration_bonus=kwargs.get('exploration_bonus', 1.0)
            )
        )
        
        # Initialize reward calculator
        self.reward_calculator = EnhancedRewardCalculator(
            info_gain_weight=kwargs.get('info_gain_weight', 0.3),
            exploration_weight=kwargs.get('exploration_weight', 0.5),
            progressive_weight=kwargs.get('progressive_weight', 0.3),
            distance_penalty_weight=kwargs.get('distance_penalty_weight', 0.2),
            belief_weight=kwargs.get('belief_weight', 0.4),
            exploration_bonus=kwargs.get('exploration_bonus', 1.0),
            progressive_factor=progressive_factor
        )
        
        self.root = POMCPNode(
            action=None,
            parent=None,
            belief=ISRSBelief(current=ISRS_STATE.RSGOOD, visited=[], location_beliefs=[], cost_expended=0),
            rng=rng
        )
        self.observation_history = []

    def _convert_to_enhanced_observation(
        self,
        obs: ISRSObservation,
        info_gain: float = 0.0,
        uncertainty: float = 0.0
    ) -> EnhancedObservation:
        """Convert basic observation to enhanced observation.
        
        Args:
            obs: Basic observation
            info_gain: Information gain from observation
            uncertainty: Uncertainty in observation
            
        Returns:
            Enhanced observation
        """
        return EnhancedObservation(
            current=obs.current,
            visited=obs.visited,
            location_states=obs.location_states,
            cost_expended=obs.cost_expended,
            obs_weight=1.0,
            info_gain=info_gain,
            uncertainty=uncertainty
        )

    def _simulate(self, state: ISRSWorldState, node: POMCPNode, depth: int) -> float:
        """Simulate a trajectory from the current state with enhanced components.
        
        Args:
            state: Current state
            node: Current node
            depth: Current depth
            
        Returns:
            Total discounted reward
        """
        if depth >= self.max_depth or self.pomdp.is_terminal(state):
            return 0.0

        # Select action using enhanced UCB
        action = self._select_action(node, state)
        if action is None:
            return 0.0

        # Get child node
        if action not in node.children:
            node.children[action] = POMCPNode(
                action=action,
                parent=node,
                belief=self._update_belief(state, action, self.pomdp.generate_o(state, action, state, self.rng)),
                rng=self.rng
            )

        # Simulate action
        next_state = self.pomdp.generate_s(state, action, self.rng)
        basic_obs = self.pomdp.generate_o(state, action, next_state, self.rng)
        
        # Calculate information gain and uncertainty
        info_gain = self._calculate_info_gain(state, action, next_state)
        uncertainty = self._calculate_uncertainty(next_state)
        
        # Convert to enhanced observation
        enhanced_obs = self._convert_to_enhanced_observation(
            basic_obs,
            info_gain=info_gain,
            uncertainty=uncertainty
        )
        
        # Apply observation widening
        enhanced_obs = self.observation_widener.add_observation(enhanced_obs)
        
        # Update belief state
        belief = self._update_belief(state, action, basic_obs)
        self.belief_cluster.add_belief(belief)
        
        # Update GP belief
        x = self.gp_belief.get_feature_vector(next_state)
        y = self._get_enhanced_reward(state, action, next_state, belief)[0]
        self.gp_belief.update(x, y)
        
        # Continue simulation
        if enhanced_obs not in node.children[action].children:
            node.children[action].children[enhanced_obs] = POMCPNode(
                action=action,
                parent=node.children[action],
                belief=belief,
                rng=self.rng
            )
            return y + self.discount_factor * self._rollout(next_state, depth + 1)
        else:
            next_node = node.children[action].children[enhanced_obs]
            value = y + self.discount_factor * self._simulate(next_state, next_node, depth + 1)
            node.children[action].value = (
                node.children[action].value * node.children[action].visits + value
            ) / (node.children[action].visits + 1)
            node.children[action].visits += 1
            return value

    def _calculate_uncertainty(self, state: ISRSWorldState) -> float:
        """Calculate uncertainty in state.
        
        Args:
            state: Current state
            
        Returns:
            Uncertainty value
        """
        # Calculate proportion of unknown states
        unknown_prop = np.mean(state.location_states == ISRS_STATE.RSUNKNOWN)
        
        # Calculate entropy of known states
        known_states = state.location_states[state.location_states != ISRS_STATE.RSUNKNOWN]
        if len(known_states) > 0:
            probs = np.array([
                np.mean(known_states == ISRS_STATE.RSGOOD),
                np.mean(known_states == ISRS_STATE.RSBAD)
            ])
            probs = probs[probs > 0]  # Remove zero probabilities
            entropy = -np.sum(probs * np.log2(probs))
        else:
            entropy = 0.0
            
        return unknown_prop + 0.5 * entropy

    def get_action(self, belief: ISRSBelief) -> MultimodalIPPAction:
        """Get action for current belief state.
        
        Args:
            belief: Current belief state
            
        Returns:
            Selected action
        """
        # Sample state from belief
        state = belief.sample_state(self.rng)
        
        # Run parallel simulations
        try:
            results = self.parallel_simulator.simulate_trajectories(
                state=state,
                num_simulations=self.num_simulations,
                get_actions=self._get_actions
            )
            
            # Process results
            for result in results:
                if result is not None:
                    action = result.action
                    node = self.root.children.get(action)
                    if node is None:
                        node = POMCPNode(
                            action=action,
                            parent=self.root,
                            belief=belief,
                            rng=self.rng
                        )
                        self.root.children[action] = node
                    node.value += result.value
                    node.visits += 1
            
            # Select best action
            return self._select_best_action(self.root)
            
        except Exception as e:
            logger.error(f"Error in get_action: {str(e)}")
            return self._get_random_action(belief)

    def _update_belief(self, state: ISRSWorldState, action: MultimodalIPPAction, observation: ISRSObservation) -> ISRSBelief:
        """Update belief state based on action and observation.
        
        Args:
            state: Current state
            action: Action taken
            observation: Observation received
            
        Returns:
            Updated belief state
        """
        # Update location beliefs based on action type
        if action.visit_location is not None:
            # Perfect observation at visited location
            new_location_beliefs = self.pomdp.env.belief_update_location_states_visit(
                state.location_beliefs,
                action.visit_location
            )
        else:
            # Sensor observation update
            new_location_beliefs = self.pomdp.env.belief_update_location_states_sensor(
                state.location_beliefs,
                observation.location_states,
                observation.current,
                action.sensing_action
            )
            
        # Create updated belief state with all required arguments
        return ISRSBelief(
            current=observation.current,
            visited=observation.visited,
            location_beliefs=new_location_beliefs,
            cost_expended=observation.cost_expended
        )

    def _get_enhanced_reward(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction,
        next_state: ISRSWorldState,
        belief: ISRSBelief
    ) -> Tuple[float, Dict[str, float]]:
        """Get enhanced reward with multiple components.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            belief: Current belief
            
        Returns:
            Total reward and component breakdown
        """
        # Base reward
        base_reward = self.pomdp.get_reward(state, action, next_state)
        
        # Information gain
        info_gain = self._calculate_info_gain(state, action, next_state)
        
        # Exploration bonus
        exploration_bonus = self._calculate_exploration_bonus(state, action)
        
        # Progressive reward
        progress_factor = self._calculate_progress_factor(state)
        
        # Distance penalty
        distance_penalty = self._calculate_distance_penalty(state, action)
        
        # Belief bonus
        belief_bonus = self._calculate_belief_bonus(belief)
        
        # Combine components with updated weights
        total_reward = (
            base_reward +
            self.info_gain_weight * info_gain +
            self.exploration_weight * exploration_bonus +
            self.progressive_weight * progress_factor -
            self.distance_penalty_weight * distance_penalty +
            self.belief_weight * belief_bonus
        )
        
        return total_reward, {
            'base': base_reward,
            'info_gain': info_gain,
            'exploration': exploration_bonus,
            'progress': progress_factor,
            'distance': distance_penalty,
            'belief': belief_bonus
        }

    def _calculate_info_gain(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction,
        next_state: ISRSWorldState
    ) -> float:
        """Calculate information gain from action.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Information gain measure
        """
        # Calculate entropy reduction
        prev_entropy = self._get_state_entropy(state)
        next_entropy = self._get_state_entropy(next_state)
        return max(0.0, prev_entropy - next_entropy)

    def _calculate_exploration_bonus(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate exploration bonus for action.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Exploration bonus
        """
        # Bonus for visiting new locations
        if action.target not in state.visited:
            return self.exploration_bonus
        
        # Reduced bonus for revisiting locations
        return self.exploration_bonus * 0.5

    def _calculate_progress_factor(self, state: ISRSWorldState) -> float:
        """Calculate progress factor.
        
        Args:
            state: Current state
            
        Returns:
            Progress factor
        """
        # Calculate progress based on visited good locations
        n_good = sum(1 for loc, val in state.location_states.items() if val == ISRS_STATE.RSGOOD)
        n_visited_good = sum(1 for loc in state.visited if state.location_states[loc] == ISRS_STATE.RSGOOD)
        return (n_visited_good / max(1, n_good)) ** self.progressive_factor

    def _calculate_distance_penalty(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Calculate distance penalty for action.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Distance penalty
        """
        # Calculate normalized distance
        max_dist = np.sqrt(2)  # Maximum possible distance in unit square
        dist = np.linalg.norm(np.array(action.target) - np.array(state.current_location))
        return dist / max_dist

    def _calculate_belief_bonus(self, belief: ISRSBelief) -> float:
        """Calculate belief bonus.
        
        Args:
            belief: Current belief
            
        Returns:
            Belief bonus
        """
        # Bonus for high confidence in good locations
        confidence = 0.0
        for loc, loc_belief in belief.location_beliefs.items():
            p_good = loc_belief.get_probability(ISRS_STATE.RSGOOD)
            confidence += p_good if p_good > 0.7 else 0.0
        return confidence / len(belief.location_beliefs)

    def _get_state_entropy(self, state: ISRSWorldState) -> float:
        """Calculate entropy of state beliefs.
        
        Args:
            state: Current state
            
        Returns:
            State entropy
        """
        entropy = 0.0
        for loc, belief in state.location_beliefs.items():
            if loc not in state.visited:
                p_good = belief.get_probability(ISRS_STATE.RSGOOD)
                p_bad = belief.get_probability(ISRS_STATE.RSBAD)
                if p_good > 0 and p_bad > 0:
                    entropy -= p_good * np.log2(p_good) + p_bad * np.log2(p_bad)
        return entropy

    def _select_action(self, node: POMCPNode, state: ISRSWorldState) -> Optional[MultimodalIPPAction]:
        """Select action using enhanced UCB.
        
        Args:
            node: Current node
            state: Current state
            
        Returns:
            Selected action or None if no valid actions
        """
        actions = self._get_actions(state)
        if not actions:
            return None

        # If some actions haven't been tried, select randomly from them
        untried_actions = [a for a in actions if a not in node.children]
        if untried_actions:
            return self.rng.choice(untried_actions)

        # Calculate action scores considering all components
        action_scores = []
        for action in actions:
            child = node.children[action]
            
            # Base value term
            if child.visits == 0:
                value_term = float('inf')
            else:
                value_term = max(child.value, 0.0)

            # Calculate information gain
            info_gain = self._estimate_information_gain(state, action)

            # Calculate exploration bonus
            exploration_bonus = self._calculate_exploration_bonus(state, action)
            
            # Calculate progressive reward
            progress_factor = self._calculate_progress_factor(state)
            
            # Combine components
            score = (
                value_term +  # Base value
                self.info_gain_weight * info_gain +  # Information gain
                self.exploration_weight * exploration_bonus +  # Exploration
                self.progressive_weight * progress_factor  # Progressive reward
            )

            # Add UCB exploration term
            if child.visits > 0 and node.visits > 0:
                score += self.exploration_constant * np.sqrt(np.log(node.visits) / child.visits)

            action_scores.append(score)

        return actions[np.argmax(action_scores)]

    def _estimate_information_gain(
        self,
        state: ISRSWorldState,
        action: MultimodalIPPAction
    ) -> float:
        """Estimate information gain for action.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Estimated information gain
        """
        if action.sensing_action is None:
            return 0.0

        # Find nearest rock for information gain calculation
        nearest_rock_dist = float('inf')
        for i, rock_state in enumerate(state.location_states):
            if rock_state in [ISRS_STATE.RSGOOD, ISRS_STATE.RSBAD]:
                dist = np.linalg.norm(
                    np.array(self.pomdp.env.location_metadata[state.current]) - 
                    np.array(self.pomdp.env.location_metadata[i])
                )
                nearest_rock_dist = min(nearest_rock_dist, dist)
        
        if nearest_rock_dist < float('inf'):
            return action.sensing_action.efficiency / (1 + nearest_rock_dist)
        return 0.0

    def _get_actions(self, state: ISRSWorldState) -> List[MultimodalIPPAction]:
        """Get available actions for state.
        
        Args:
            state: Current state
            
        Returns:
            List of available actions
        """
        return self.pomdp.get_actions(state)

    def _get_random_action(self, belief: ISRSBelief) -> MultimodalIPPAction:
        """Get random action from available actions.
        
        Args:
            belief: Current belief state
            
        Returns:
            Random action
        """
        state = belief.sample_state(self.rng)
        actions = self._get_actions(state)
        return self.rng.choice(actions)

    def _rollout(self, state: ISRSWorldState, depth: int) -> float:
        """Perform rollout from current state with enhanced components.
        
        Args:
            state: Current state
            depth: Current depth
            
        Returns:
            Total discounted reward
        """
        if depth >= self.max_depth or self.pomdp.is_terminal(state):
            return 0.0
            
        action = self._get_random_action(ISRSBelief(state))
        next_state = self.pomdp.generate_s(state, action, self.rng)
        
        # Get GP belief value
        gp_value = self.gp_belief.get_belief(next_state)
        
        # Calculate enhanced reward
        reward = self._get_enhanced_reward(state, action, next_state, ISRSBelief(state))[0]
        
        # Combine reward with GP belief
        combined_reward = 0.7 * reward + 0.3 * gp_value
        
        return combined_reward + self.discount_factor * self._rollout(next_state, depth + 1)

    def _select_best_action(self, node: POMCPNode) -> MultimodalIPPAction:
        """Select best action based on enhanced criteria.
        
        Args:
            node: Current node
            
        Returns:
            Best action
        """
        if not node.children:
            return self._get_random_action(node.belief)
            
        # Select action with highest average value
        return max(
            node.children.values(),
            key=lambda n: n.value / n.visits if n.visits > 0 else 0
        ).action

class POMCPDPWPolicy(EnhancedPOMCPPolicy):
    """POMCP with Double Progressive Widening.
    
    This policy implements Double Progressive Widening (DPW) for both actions and observations:
    - Action widening: k_a * N^α_a controls the number of available actions
    - Observation widening: k_o * N^α_o controls the number of observations
    
    Attributes:
        pomdp: POMDP model
        rng: Random number generator
        num_simulations: Number of simulations
        exploration_constant: UCB exploration constant
        k_action: Action widening coefficient
        alpha_action: Action widening exponent
        k_observation: Observation widening coefficient
        alpha_observation: Observation widening exponent
    """
    def __init__(
        self,
        pomdp: Any,
        rng: np.random.Generator,
        num_simulations: int = 1000,
        exploration_constant: float = 1.0,
        k_action: float = 0.5,
        alpha_action: float = 0.5,
        k_observation: float = 0.5,
        alpha_observation: float = 0.5,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize POMCPDPW policy.
        
        Args:
            pomdp: POMDP model
            rng: Random number generator
            num_simulations: Number of simulations
            exploration_constant: UCB exploration constant
            k_action: Action widening coefficient
            alpha_action: Action widening exponent
            k_observation: Observation widening coefficient
            alpha_observation: Observation widening exponent
            **kwargs: Additional arguments
        """
        super().__init__(
            pomdp=pomdp,
            rng=rng,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            **kwargs
        )
        self.k_action = k_action
        self.alpha_action = alpha_action
        self.k_observation = k_observation
        self.alpha_observation = alpha_observation
        
        # Initialize parallel simulator with updated config
        self.parallel_simulator = ParallelSimulator(
            pomdp=pomdp,
            rng=rng,
            config=SimulationConfig(
                max_depth=kwargs.get('max_depth', 5),
                discount_factor=kwargs.get('discount_factor', 0.95),
                num_processes=kwargs.get('num_processes', None),
                batch_size=kwargs.get('batch_size', 100),
                rollout_strategy=kwargs.get('rollout_strategy', RolloutStrategy.RANDOM),
                show_progress=kwargs.get('show_progress', True),
                exploration_constant=exploration_constant,
                progressive_factor=kwargs.get('progressive_factor', 0.5),
                alpha_action=alpha_action,
                alpha_observation=alpha_observation,
                width=kwargs.get('width', 0.1),
                similarity_threshold=kwargs.get('similarity_threshold', 0.6),
                max_width=kwargs.get('max_width', 0.5),
                min_width=kwargs.get('min_width', 0.01),
                width_decay=kwargs.get('width_decay', 0.95),
                belief_similarity_threshold=kwargs.get('belief_similarity_threshold', 0.7),
                max_belief_size=kwargs.get('max_belief_size', 100),
                gp_noise_level=kwargs.get('gp_noise_level', 1e-4),
                gp_n_restarts=kwargs.get('gp_n_restarts', 10),
                info_gain_weight=kwargs.get('info_gain_weight', 0.3),
                exploration_weight=kwargs.get('exploration_weight', 0.5),
                progressive_weight=kwargs.get('progressive_weight', 0.3),
                distance_penalty_weight=kwargs.get('distance_penalty_weight', 0.2),
                belief_weight=kwargs.get('belief_weight', 0.4),
                exploration_bonus=kwargs.get('exploration_bonus', 1.0)
            )
        )
        
        # Store available actions for parallel simulation
        self.available_actions = []

    def _get_actions_for_simulation(self, state: ISRSWorldState) -> List[MultimodalIPPAction]:
        """Get available actions for simulation.
        
        Args:
            state: Current state
            
        Returns:
            List of available actions
        """
        return self.available_actions

    def select_action(
        self,
        belief: ISRSBelief,
        available_actions: List[MultimodalIPPAction]
    ) -> MultimodalIPPAction:
        """Select action using DPW-POMCP.
        
        Args:
            belief: Current belief state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        if not available_actions:
            raise ValueError("No available actions")
            
        # Store available actions for parallel simulation
        self.available_actions = available_actions
            
        # Sample state from belief
        state = belief.sample_state(self.rng)
        
        # Run parallel simulations
        try:
            results, stats = self.parallel_simulator.simulate_trajectories(
                state=state,
                num_simulations=self.num_simulations,
                get_actions=self._get_actions_for_simulation
            )
            
            # Find best action based on average value
            action_values = {}
            action_counts = {}
            
            for result in results:
                action = result.action
                if action not in action_values:
                    action_values[action] = 0.0
                    action_counts[action] = 0
                action_values[action] += result.value
                action_counts[action] += 1
            
            # Calculate UCB values
            ucb_values = {}
            total_sims = len(results)
            
            for action in action_values:
                avg_value = action_values[action] / action_counts[action]
                exploration = np.sqrt(
                    2 * np.log(total_sims) / action_counts[action]
                )
                ucb_values[action] = avg_value + self.exploration_constant * exploration
            
            # Select action with highest UCB value
            best_action = max(ucb_values.items(), key=lambda x: x[1])[0]
            return best_action
            
        except Exception as e:
            logger.error(f"Error in select_action: {str(e)}")
            return self.rng.choice(available_actions)

def get_pomcp_dpw_policy(pomdp: ISRSPOMDP, rng: np.random.RandomState) -> POMCPDPWPolicy:
    """Get POMCP policy with Double Progressive Widening."""
    return POMCPDPWPolicy(
        pomdp=pomdp,
        rng=rng,
        num_simulations=200,
        exploration_constant=2.0,
        k_action=0.8,
        alpha_action=0.6,
        k_observation=0.8,
        alpha_observation=0.6,
        info_gain_weight=0.4,
        exploration_weight=0.6,
        progressive_weight=0.4,
        distance_penalty_weight=0.1,
        belief_weight=0.5,
        exploration_bonus=2.0,
        progressive_factor=0.95,
        belief_similarity_threshold=0.7,
        max_depth=10,
        discount_factor=0.99
    )

def get_pomcp_basic_policy(pomdp: ISRSPOMDP, rng: np.random.RandomState) -> EnhancedPOMCPPolicy:
    """Get basic POMCP policy."""
    return EnhancedPOMCPPolicy(
        pomdp=pomdp,
        rng=rng,
        max_depth=5,
        num_simulations=100,
        exploration_constant=1.0,
        discount_factor=0.95
    )

def get_pomcp_gcb_policy(pomdp: ISRSPOMDP, rng: np.random.RandomState) -> EnhancedPOMCPPolicy:
    """Get POMCP policy with Goal-Conditioned Belief."""
    return EnhancedPOMCPPolicy(
        pomdp=pomdp,
        rng=rng,
        max_depth=5,
        num_simulations=100,
        exploration_constant=1.0,
        discount_factor=0.95,
        info_gain_weight=0.3,
        exploration_weight=0.5,
        progressive_weight=0.3,
        observation_threshold=0.1,
        k_action=0.5,
        alpha_action=0.5,
        k_observation=0.5,
        alpha_observation=0.5
    )

def get_pomcpow_policy(pomdp: ISRSPOMDP, rng: np.random.RandomState) -> EnhancedPOMCPPolicy:
    """Get POMCP policy with Observation Widening."""
    return EnhancedPOMCPPolicy(
        pomdp=pomdp,
        rng=rng,
        max_depth=5,
        num_simulations=100,
        exploration_constant=1.0,
        discount_factor=0.95,
        info_gain_weight=0.3,
        exploration_weight=0.5,
        progressive_weight=0.3,
        observation_threshold=0.1,
        k_action=0.5,
        alpha_action=0.5,
        k_observation=0.5,
        alpha_observation=0.5
    )

class InformationBasedPOMCP(BasePolicy):
    """Information-based POMCP policy.
    
    Attributes:
        pomdp: POMDP model
        rng: Random number generator
        num_simulations: Number of simulations
        max_depth: Maximum search depth
        exploration_constant: UCB exploration constant
        info_weight: Weight for information gain
    """
    def __init__(
        self,
        pomdp: Any,
        rng: np.random.Generator,
        num_simulations: int = 1000,
        max_depth: int = 50,
        exploration_constant: float = 1.0,
        info_weight: float = 0.5,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize information-based POMCP policy.
        
        Args:
            pomdp: POMDP model
            rng: Random number generator
            num_simulations: Number of simulations
            max_depth: Maximum search depth
            exploration_constant: UCB exploration constant
            info_weight: Weight for information gain
            **kwargs: Additional arguments
        """
        super().__init__(
            pomdp=pomdp,
            rng=rng,
            num_simulations=num_simulations,
            max_depth=max_depth,
            exploration_constant=exploration_constant
        )
        self.info_weight = info_weight

class NaivePolicy(BasePolicy):
    """Naive policy that selects actions randomly.
    
    Attributes:
        pomdp: POMDP model
        rng: Random number generator
    """
    def __init__(
        self,
        pomdp: Any,
        rng: np.random.Generator,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize naive policy.
        
        Args:
            pomdp: POMDP model
            rng: Random number generator
            **kwargs: Additional arguments
        """
        super().__init__(
            pomdp=pomdp,
            rng=rng,
            num_simulations=1,
            max_depth=1,
            exploration_constant=0.0
        )

class RandomPolicy(BasePolicy):
    """Random policy for ISRS environment."""
    
    def select_action(
        self,
        belief: ISRSBelief,
        available_actions: List[MultimodalIPPAction]
    ) -> MultimodalIPPAction:
        """Select random action.
        
        Args:
            belief: Current belief state
            available_actions: List of available actions
            
        Returns:
            Randomly selected action
        """
        return np.random.choice(available_actions)

class GreedyPolicy(BasePolicy):
    """Greedy policy for ISRS environment."""
    
    def __init__(self, pomdp: 'ISRSPOMDP', rng: np.random.RandomState) -> None:
        """Initialize policy.
        
        Args:
            pomdp: POMDP instance
            rng: Random number generator
        """
        self.pomdp = pomdp
        self.rng = rng
        
    def select_action(
        self,
        belief: ISRSBelief,
        available_actions: List[MultimodalIPPAction]
    ) -> MultimodalIPPAction:
        """Select action using greedy strategy.
        
        Args:
            belief: Current belief state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        # Sample state from belief
        state = belief.sample_state(self.rng)
        
        # Evaluate each action
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            # Simulate action
            next_state, _, reward, _, next_belief = self.pomdp.step(action)
            
            # Calculate value
            value = reward
            
            # Update best action
            if value > best_value:
                best_action = action
                best_value = value
                
        return best_action

class POMCPPolicy(BasePolicy):
    """POMCP policy for ISRS environment."""
    
    def __init__(
        self,
        pomdp: 'ISRSPOMDP',
        rng: np.random.RandomState,
        exploration_constant: float = 1.0,
        discount_factor: float = 0.95,
        max_depth: int = 50,
        num_simulations: int = 1000
    ) -> None:
        """Initialize policy.
        
        Args:
            pomdp: POMDP instance
            rng: Random number generator
            exploration_constant: UCB exploration constant
            discount_factor: Discount factor for future rewards
            max_depth: Maximum search depth
            num_simulations: Number of simulations per action selection
        """
        self.pomdp = pomdp
        self.rng = rng
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        
    def select_action(
        self,
        belief: ISRSBelief,
        available_actions: List[MultimodalIPPAction]
    ) -> MultimodalIPPAction:
        """Select action using POMCP.
        
        Args:
            belief: Current belief state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        # Sample state from belief
        state = belief.sample_state(self.rng)
        
        # Initialize root node
        root = POMCPNode(
            action=None,
            parent=None,
            belief=belief,
            rng=self.rng
        )
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Sample state from belief
            sampled_state = belief.sample_state(self.rng)
            
            # Simulate from root
            self._simulate(
                node=root,
                state=sampled_state,
                depth=0
            )
            
        # Select best action
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            # Get child node for action
            child = root.get_child(action)
            if child is None:
                continue
                
            # Calculate value
            value = child.value / max(1, child.visits)
            
            # Update best action
            if value > best_value:
                best_action = action
                best_value = value
                
        return best_action or available_actions[0]
        
    def _simulate(
        self,
        node: 'POMCPNode',
        state: ISRSWorldState,
        depth: int
    ) -> float:
        """Simulate from node.
        
        Args:
            node: Current node
            state: Current state
            depth: Current depth
            
        Returns:
            Discounted reward
        """
        # Check termination
        if depth >= self.max_depth or self.pomdp.is_terminal(state):
            return 0.0
            
        # Get available actions
        available_actions = self.pomdp.get_actions(state)
        
        # Select action using UCB
        action = self._select_action_ucb(node, available_actions)
        
        # Get child node
        child = node.get_child(action)
        
        # If child doesn't exist, expand
        if child is None:
            # Simulate action
            next_state, observation, reward, done, next_belief = self.pomdp.step(action)
            
            # Create child node
            child = POMCPNode(
                action=action,
                parent=node,
                belief=next_belief,
                rng=self.rng
            )
            node.add_child(action, child)
            
            # Get value from rollout
            value = reward + self.discount_factor * self._rollout(next_state, depth + 1)
        else:
            # Simulate action
            next_state, observation, reward, done, next_belief = self.pomdp.step(action)
            
            # Get value from recursion
            value = reward + self.discount_factor * self._simulate(child, next_state, depth + 1)
            
        # Update statistics
        node.visits += 1
        node.value += value
        
        return value
        
    def _rollout(self, state: ISRSWorldState, depth: int) -> float:
        """Perform rollout from state.
        
        Args:
            state: Current state
            depth: Current depth
            
        Returns:
            Discounted reward
        """
        # Check termination
        if depth >= self.max_depth or self.pomdp.is_terminal(state):
            return 0.0
            
        # Get available actions
        available_actions = self.pomdp.get_actions(state)
        
        # Select random action
        action = available_actions[self.rng.randint(len(available_actions))]
        
        # Simulate action
        next_state, _, reward, _, _ = self.pomdp.step(action)
        
        # Recurse
        return reward + self.discount_factor * self._rollout(next_state, depth + 1)
        
    def _select_action_ucb(
        self,
        node: 'POMCPNode',
        available_actions: List[MultimodalIPPAction]
    ) -> MultimodalIPPAction:
        """Select action using UCB.
        
        Args:
            node: Current node
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        # If not all actions tried, select untried action
        untried_actions = [
            action for action in available_actions
            if node.get_child(action) is None
        ]
        if untried_actions:
            return untried_actions[self.rng.randint(len(untried_actions))]
            
        # Select action using UCB
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            child = node.get_child(action)
            if child is None:
                continue
                
            # Calculate UCB value
            exploitation = child.value / max(1, child.visits)
            exploration = self.exploration_constant * np.sqrt(
                np.log(node.visits) / max(1, child.visits)
            )
            value = exploitation + exploration
            
            # Update best action
            if value > best_value:
                best_action = action
                best_value = value
                
        return best_action or available_actions[0]

class InformationSeekingPolicy(BasePolicy):
    """Information-seeking policy for ISRS environment."""
    
    def __init__(self, pomdp: 'ISRSPOMDP', rng: np.random.RandomState) -> None:
        """Initialize policy.
        
        Args:
            pomdp: POMDP instance
            rng: Random number generator
        """
        super().__init__(pomdp=pomdp, rng=rng)
        self.env = pomdp.env
    
    def select_action(
        self,
        belief: ISRSBelief,
        available_actions: List[MultimodalIPPAction]
    ) -> MultimodalIPPAction:
        """Select action that maximizes information gain.
        
        Args:
            belief: Current belief state
            available_actions: List of available actions
            
        Returns:
            Action that maximizes information gain
        """
        best_action = None
        best_info_gain = float('-inf')
        
        for action in available_actions:
            # Simulate action
            next_state, _, _, _, next_belief = self.env.step(action)
            
            # Calculate information gain
            current_rmse, current_trace = belief.get_metrics(
                belief.sample_state(np.random.RandomState())
            )
            next_rmse, next_trace = next_belief.get_metrics(
                next_belief.sample_state(np.random.RandomState())
            )
            
            info_gain = (current_trace - next_trace) / max(1.0, current_trace)
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_action = action
                
        return best_action
