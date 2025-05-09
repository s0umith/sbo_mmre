"""Enhanced policy types for ISRS environment."""

from typing import Optional, Dict, Any
import numpy as np

from .policies import EnhancedPOMCPPolicy, POMCPNode
from .states import ISRSWorldState
from .actions import MultimodalIPPAction
from .pomdp import ISRSPOMDP

class POMCPDPWPolicy(EnhancedPOMCPPolicy):
    """POMCP with Double Progressive Widening for both actions and observations.
    
    This policy extends the enhanced POMCP with:
    - Double progressive widening for both actions and observations
    - Enhanced observation handling with GP features
    - Parallel simulation support
    - Advanced reward structure
    """
    
    def __init__(
        self,
        pomdp: ISRSPOMDP,
        rng: np.random.RandomState,
        max_depth: int = 5,
        num_simulations: int = 100,
        exploration_constant: float = 1.0,
        discount_factor: float = 0.95,
        info_gain_weight: float = 0.3,
        exploration_weight: float = 0.5,
        progressive_weight: float = 0.3,
        observation_threshold: float = 0.1,
        k_action: float = 0.5,
        alpha_action: float = 0.5,
        k_observation: float = 0.5,
        alpha_observation: float = 0.5,
        n_belief_clusters: int = 5,
        belief_similarity_threshold: float = 0.8,
        gp_noise_level: float = 1e-4,
        gp_n_restarts: int = 10,
        num_processes: Optional[int] = None,
        distance_penalty_weight: float = 0.2,
        belief_weight: float = 0.4,
        exploration_bonus: float = 1.0,
        progressive_factor: float = 0.9
    ) -> None:
        """Initialize POMCPDPW policy.
        
        Args:
            pomdp: POMDP instance
            rng: Random number generator
            max_depth: Maximum depth of search tree
            num_simulations: Number of simulations per action selection
            exploration_constant: UCB exploration constant
            discount_factor: Discount factor for future rewards
            info_gain_weight: Weight for information gain in reward
            exploration_weight: Weight for exploration bonus
            progressive_weight: Weight for progressive rewards
            observation_threshold: Threshold for observation similarity
            k_action: Action widening parameter
            alpha_action: Action widening exponent
            k_observation: Observation widening parameter
            alpha_observation: Observation widening exponent
            n_belief_clusters: Number of belief state clusters
            belief_similarity_threshold: Threshold for belief state similarity
            gp_noise_level: Noise level for Gaussian Process
            gp_n_restarts: Number of restarts for GP optimization
            num_processes: Number of processes for parallel simulation
            distance_penalty_weight: Weight for distance penalty
            belief_weight: Weight for belief bonus
            exploration_bonus: Base exploration bonus
            progressive_factor: Base progressive factor
        """
        super().__init__(
            pomdp=pomdp,
            rng=rng,
            max_depth=max_depth,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,
            info_gain_weight=info_gain_weight,
            exploration_weight=exploration_weight,
            progressive_weight=progressive_weight,
            observation_threshold=observation_threshold,
            k_action=k_action,
            alpha_action=alpha_action,
            k_observation=k_observation,
            alpha_observation=alpha_observation,
            n_belief_clusters=n_belief_clusters,
            belief_similarity_threshold=belief_similarity_threshold,
            gp_noise_level=gp_noise_level,
            gp_n_restarts=gp_n_restarts,
            num_processes=num_processes,
            distance_penalty_weight=distance_penalty_weight,
            belief_weight=belief_weight,
            exploration_bonus=exploration_bonus,
            progressive_factor=progressive_factor
        )

class InformationBasedPOMCP(EnhancedPOMCPPolicy):
    """POMCP with information-based exploration.
    
    This policy extends the enhanced POMCP with:
    - Information gain-based action selection
    - Enhanced observation handling with GP features
    - Parallel simulation support
    - Advanced reward structure
    """
    
    def __init__(
        self,
        pomdp: ISRSPOMDP,
        rng: np.random.RandomState,
        max_depth: int = 5,
        num_simulations: int = 100,
        exploration_constant: float = 1.0,
        discount_factor: float = 0.95,
        info_gain_weight: float = 0.3,
        exploration_weight: float = 0.5,
        progressive_weight: float = 0.3,
        observation_threshold: float = 0.1,
        k_action: float = 0.5,
        alpha_action: float = 0.5,
        k_observation: float = 0.5,
        alpha_observation: float = 0.5,
        n_belief_clusters: int = 5,
        belief_similarity_threshold: float = 0.8,
        gp_noise_level: float = 1e-4,
        gp_n_restarts: int = 10,
        num_processes: Optional[int] = None,
        distance_penalty_weight: float = 0.2,
        belief_weight: float = 0.4,
        exploration_bonus: float = 1.0,
        progressive_factor: float = 0.9,
        information_threshold: float = 0.1,
        exploration_ratio: float = 0.3
    ) -> None:
        """Initialize information-based POMCP policy.
        
        Args:
            pomdp: POMDP instance
            rng: Random number generator
            max_depth: Maximum depth of search tree
            num_simulations: Number of simulations per action selection
            exploration_constant: UCB exploration constant
            discount_factor: Discount factor for future rewards
            info_gain_weight: Weight for information gain in reward
            exploration_weight: Weight for exploration bonus
            progressive_weight: Weight for progressive rewards
            observation_threshold: Threshold for observation similarity
            k_action: Action widening parameter
            alpha_action: Action widening exponent
            k_observation: Observation widening parameter
            alpha_observation: Observation widening exponent
            n_belief_clusters: Number of belief state clusters
            belief_similarity_threshold: Threshold for belief state similarity
            gp_noise_level: Noise level for Gaussian Process
            gp_n_restarts: Number of restarts for GP optimization
            num_processes: Number of processes for parallel simulation
            distance_penalty_weight: Weight for distance penalty
            belief_weight: Weight for belief bonus
            exploration_bonus: Base exploration bonus
            progressive_factor: Base progressive factor
            information_threshold: Threshold for information-based action selection
            exploration_ratio: Ratio of exploration to exploitation
        """
        super().__init__(
            pomdp=pomdp,
            rng=rng,
            max_depth=max_depth,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,
            info_gain_weight=info_gain_weight,
            exploration_weight=exploration_weight,
            progressive_weight=progressive_weight,
            observation_threshold=observation_threshold,
            k_action=k_action,
            alpha_action=alpha_action,
            k_observation=k_observation,
            alpha_observation=alpha_observation,
            n_belief_clusters=n_belief_clusters,
            belief_similarity_threshold=belief_similarity_threshold,
            gp_noise_level=gp_noise_level,
            gp_n_restarts=gp_n_restarts,
            num_processes=num_processes,
            distance_penalty_weight=distance_penalty_weight,
            belief_weight=belief_weight,
            exploration_bonus=exploration_bonus,
            progressive_factor=progressive_factor
        )
        self.information_threshold = information_threshold
        self.exploration_ratio = exploration_ratio

    def _select_action(self, node: POMCPNode, state: ISRSWorldState) -> Optional[MultimodalIPPAction]:
        """Select action based on information gain.
        
        Args:
            node: Current node
            state: Current state
            
        Returns:
            Selected action or None if no valid actions
        """
        actions = self._get_actions(state)
        if not actions:
            return None
            
        # Calculate information gain for each action
        info_gains = []
        for action in actions:
            info_gain = self._estimate_information_gain(state, action)
            info_gains.append(info_gain)
            
        # Select action with highest information gain
        max_gain = max(info_gains)
        if max_gain > self.information_threshold:
            return actions[np.argmax(info_gains)]
        else:
            # Fall back to UCB selection
            return super()._select_action(node, state)

class NaivePolicy(EnhancedPOMCPPolicy):
    """Naive policy with basic exploration.
    
    This policy extends the enhanced POMCP with:
    - Basic exploration strategy
    - Simplified observation handling
    - Parallel simulation support
    - Basic reward structure
    """
    
    def __init__(
        self,
        pomdp: ISRSPOMDP,
        rng: np.random.RandomState,
        max_depth: int = 5,
        num_simulations: int = 100,
        exploration_constant: float = 1.0,
        discount_factor: float = 0.95,
        num_processes: Optional[int] = None
    ) -> None:
        """Initialize naive policy.
        
        Args:
            pomdp: POMDP instance
            rng: Random number generator
            max_depth: Maximum depth of search tree
            num_simulations: Number of simulations per action selection
            exploration_constant: UCB exploration constant
            discount_factor: Discount factor for future rewards
            num_processes: Number of processes for parallel simulation
        """
        super().__init__(
            pomdp=pomdp,
            rng=rng,
            max_depth=max_depth,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,
            info_gain_weight=0.0,
            exploration_weight=0.0,
            progressive_weight=0.0,
            observation_threshold=1.0,
            k_action=1.0,
            alpha_action=1.0,
            k_observation=1.0,
            alpha_observation=1.0,
            n_belief_clusters=1,
            belief_similarity_threshold=1.0,
            gp_noise_level=1e-4,
            gp_n_restarts=10,
            num_processes=num_processes,
            distance_penalty_weight=0.0,
            belief_weight=0.0,
            exploration_bonus=0.0,
            progressive_factor=1.0
        )

class POMCPDPWPolicyType(EnhancedPOMCPPolicy):
    """POMCPDPW policy type."""
    def __init__(
        self,
        pomdp: Any,
        rng: np.random.Generator,
        num_simulations: int = 1000,
        max_depth: int = 50,
        exploration_constant: float = 1.0,
        width: float = 0.1,
        similarity_threshold: float = 0.6,
        max_width: float = 0.5,
        min_width: float = 0.01,
        width_decay: float = 0.95,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize POMCPDPW policy type.
        
        Args:
            pomdp: POMDP model
            rng: Random number generator
            num_simulations: Number of simulations
            max_depth: Maximum search depth
            exploration_constant: UCB exploration constant
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
            exploration_constant=exploration_constant,
            width=width,
            similarity_threshold=similarity_threshold,
            max_width=max_width,
            min_width=min_width,
            width_decay=width_decay,
            **kwargs
        )

def get_policy(
    policy_type: str,
    pomdp: Any,
    rng: np.random.Generator,
    **kwargs: Dict[str, Any]
) -> Any:
    """Get policy instance based on type.
    
    Args:
        policy_type: Type of policy to create
        pomdp: POMDP model
        rng: Random number generator
        **kwargs: Additional arguments for policy
        
    Returns:
        Policy instance
    """
    policy_map = {
        'pomcpdpw': POMCPDPWPolicyType,
        'information_based': InformationBasedPOMCP,
        'naive': NaivePolicy
    }
    
    if policy_type not in policy_map:
        raise ValueError(f'Unknown policy type: {policy_type}')
        
    # Set default parameters based on policy type
    default_params = {
        'pomcpdpw': {
            'num_simulations': 1000,
            'max_depth': 50,
            'exploration_constant': 1.0,
            'width': 0.1,
            'similarity_threshold': 0.6,
            'max_width': 0.5,
            'min_width': 0.01,
            'width_decay': 0.95
        },
        'information_based': {
            'num_simulations': 100,
            'max_depth': 5,
            'exploration_constant': 1.0,
            'discount_factor': 0.95,
            'info_gain_weight': 0.3,
            'exploration_weight': 0.5,
            'progressive_weight': 0.3,
            'observation_threshold': 0.1,
            'k_action': 0.5,
            'alpha_action': 0.5,
            'k_observation': 0.5,
            'alpha_observation': 0.5,
            'n_belief_clusters': 5,
            'belief_similarity_threshold': 0.8,
            'gp_noise_level': 1e-4,
            'gp_n_restarts': 10,
            'distance_penalty_weight': 0.2,
            'belief_weight': 0.4,
            'exploration_bonus': 1.0,
            'progressive_factor': 0.9,
            'information_threshold': 0.1,
            'exploration_ratio': 0.3
        },
        'naive': {
            'num_simulations': 100,
            'max_depth': 5,
            'exploration_constant': 1.0,
            'discount_factor': 0.95
        }
    }
    
    # Update with provided kwargs
    params = {**default_params[policy_type], **kwargs}
    
    return policy_map[policy_type](pomdp=pomdp, rng=rng, **params) 