"""Parallel simulator for ISRS POMDP with enhanced batch processing capabilities."""

from typing import List, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass
from loguru import logger
import multiprocessing as mp
from enum import Enum
import time
from tqdm import tqdm

# Local imports
from .states import ISRSWorldState, ISRSObservation, ISRSBelief
from .actions import MultimodalIPPAction
from .pomdp import ISRSPOMDP
from .belief import ISRSBelief

# Global variables for multiprocessing
_global_pomdp = None
_global_get_actions = None
_global_config = None

class RolloutStrategy(Enum):
    """Available rollout strategies."""
    RANDOM = "random"
    GREEDY = "greedy"
    HEURISTIC = "heuristic"

@dataclass
class SimulationConfig:
    """Configuration for simulation runs.
    
    Attributes:
        max_depth: Maximum depth of simulations
        discount_factor: Discount factor for rewards
        num_processes: Number of processes to use
        batch_size: Size of simulation batches
        rollout_strategy: Strategy to use for rollouts
        show_progress: Whether to show progress bar
        exploration_constant: UCB exploration constant
        progressive_factor: Progressive widening factor
        alpha_action: Action widening exponent
        alpha_observation: Observation widening exponent
        width: Initial width for observation widening
        similarity_threshold: Threshold for observation similarity
        max_width: Maximum width for observation widening
        min_width: Minimum width for observation widening
        width_decay: Width decay factor
        belief_similarity_threshold: Threshold for belief similarity
        max_belief_size: Maximum size of belief cluster
        gp_noise_level: Noise level for GP regression
        gp_n_restarts: Number of restarts for GP optimization
        info_gain_weight: Weight for information gain
        exploration_weight: Weight for exploration
        progressive_weight: Weight for progressive reward
        distance_penalty_weight: Weight for distance penalty
        belief_weight: Weight for belief reward
        exploration_bonus: Bonus for exploration
    """
    max_depth: int = 5
    discount_factor: float = 0.95
    num_processes: Optional[int] = None
    batch_size: int = 100
    rollout_strategy: RolloutStrategy = RolloutStrategy.RANDOM
    show_progress: bool = True
    exploration_constant: float = 1.0
    progressive_factor: float = 0.5
    alpha_action: float = 0.5
    alpha_observation: float = 0.5
    width: float = 0.1
    similarity_threshold: float = 0.6
    max_width: float = 0.5
    min_width: float = 0.01
    width_decay: float = 0.95
    belief_similarity_threshold: float = 0.7
    max_belief_size: int = 100
    gp_noise_level: float = 1e-4
    gp_n_restarts: int = 10
    info_gain_weight: float = 0.3
    exploration_weight: float = 0.5
    progressive_weight: float = 0.3
    distance_penalty_weight: float = 0.2
    belief_weight: float = 0.4
    exploration_bonus: float = 1.0

@dataclass
class SimulationResult:
    """Result of a single simulation.
    
    Attributes:
        action: Selected action
        value: Total discounted value
        observation: Resulting observation
        belief: Updated belief state
        depth: Depth reached in simulation
        computation_time: Time taken for simulation
    """
    action: MultimodalIPPAction
    value: float
    observation: ISRSObservation
    belief: ISRSBelief
    depth: int
    computation_time: float

@dataclass
class BatchStatistics:
    """Statistics for a batch of simulations.
    
    Attributes:
        mean_value: Mean value across simulations
        std_value: Standard deviation of values
        mean_depth: Mean depth reached
        total_time: Total computation time
        num_simulations: Number of simulations in batch
    """
    mean_value: float
    std_value: float
    mean_depth: float
    total_time: float
    num_simulations: int

def _init_worker(pomdp: ISRSPOMDP, get_actions: Callable, config: SimulationConfig) -> None:
    """Initialize worker process with global variables."""
    global _global_pomdp, _global_get_actions, _global_config
    _global_pomdp = pomdp
    _global_get_actions = get_actions
    _global_config = config

def _select_action(
    actions: List[MultimodalIPPAction],
    state: ISRSWorldState,
    rng: np.random.RandomState
) -> MultimodalIPPAction:
    """Select action based on rollout strategy."""
    if _global_config.rollout_strategy == RolloutStrategy.RANDOM:
        return rng.choice(actions)
    elif _global_config.rollout_strategy == RolloutStrategy.GREEDY:
        values = [_global_pomdp.get_reward(state, a, _global_pomdp.generate_s(state, a, rng))[0] for a in actions]
        return actions[np.argmax(values)]
    else:  # HEURISTIC
        return min(actions, key=lambda a: a.cost)

def _rollout(
    state: ISRSWorldState,
    depth: int,
    rng: np.random.RandomState
) -> Tuple[float, int]:
    """Perform rollout from state."""
    if depth >= _global_config.max_depth or _global_pomdp.is_terminal(state):
        return 0.0, depth
        
    actions = _global_get_actions(state)
    if not actions:
        return 0.0, depth
        
    action = _select_action(actions, state, rng)
    next_state = _global_pomdp.generate_s(state, action, rng)
    reward, _ = _global_pomdp.get_reward(state, action, next_state)
    
    future_reward, max_depth = _rollout(next_state, depth + 1, rng)
    return reward + _global_config.discount_factor * future_reward, max_depth

def _simulate_worker(seed: int, state: ISRSWorldState) -> Optional[SimulationResult]:
    """Worker function for parallel simulation."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    try:
        # Get available actions
        actions = _global_get_actions(state)
        if not actions:
            return None
            
        # Select action based on rollout strategy
        action = _select_action(actions, state, rng)
        
        # Simulate action
        next_state = _global_pomdp.generate_s(state, action, rng)
        observation = _global_pomdp.generate_o(state, action, next_state, rng)
        reward, _ = _global_pomdp.get_reward(state, action, next_state)
        
        # Create belief state using environment's initialization
        belief = _global_pomdp.env._initialize_belief_state()
        belief.current = next_state.current
        belief.visited = next_state.visited.copy()
        belief.cost_expended = next_state.cost_expended
        
        # Rollout from next state
        future_reward, depth = _rollout(next_state, 1, rng)
        
        # Calculate total value
        value = reward + _global_config.discount_factor * future_reward
        
        return SimulationResult(
            action=action,
            value=value,
            observation=observation,
            belief=belief,
            depth=depth,
            computation_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        return None

class ParallelSimulator:
    """Enhanced parallel simulator for ISRS POMDP with batch processing."""
    
    def __init__(
        self,
        pomdp: ISRSPOMDP,
        rng: np.random.RandomState,
        config: Optional[SimulationConfig] = None
    ) -> None:
        """Initialize parallel simulator.
        
        Args:
            pomdp: POMDP instance
            rng: Random number generator
            config: Simulation configuration
        """
        self.pomdp = pomdp
        self.rng = rng
        self.config = config or SimulationConfig()
        self.num_processes = (
            self.config.num_processes 
            if self.config.num_processes is not None 
            else max(1, mp.cpu_count() - 1)
        )
        
        logger.info(
            f"Initialized parallel simulator with {self.num_processes} processes, "
            f"batch size {self.config.batch_size}, "
            f"rollout strategy {self.config.rollout_strategy.value}"
        )

    def simulate_trajectories(
        self,
        state: ISRSWorldState,
        num_simulations: int,
        get_actions: Callable[[ISRSWorldState], List[MultimodalIPPAction]]
    ) -> Tuple[List[SimulationResult], BatchStatistics]:
        """Run multiple simulations in parallel with batch processing.
        
        Args:
            state: Initial state
            num_simulations: Number of simulations to run
            get_actions: Function to get available actions
            
        Returns:
            Tuple of (simulation results, batch statistics)
        """
        start_time = time.time()
        results: List[SimulationResult] = []
        
        try:
            # Create batches
            num_batches = (num_simulations + self.config.batch_size - 1) // self.config.batch_size
            
            # Initialize process pool with worker initialization
            with ProcessPoolExecutor(
                max_workers=self.num_processes,
                initializer=_init_worker,
                initargs=(self.pomdp, get_actions, self.config)
            ) as executor:
                with tqdm(total=num_simulations, disable=not self.config.show_progress) as pbar:
                    for batch_idx in range(num_batches):
                        batch_size = min(
                            self.config.batch_size,
                            num_simulations - batch_idx * self.config.batch_size
                        )
                        
                        # Generate seeds for this batch
                        seeds = [self.rng.randint(0, 2**32) for _ in range(batch_size)]
                        
                        # Submit batch jobs
                        futures = [
                            executor.submit(_simulate_worker, seed, state)
                            for seed in seeds
                        ]
                        
                        # Collect results
                        batch_results = []
                        for future in as_completed(futures):
                            result = future.result()
                            if result is not None:
                                batch_results.append(result)
                        
                        results.extend(batch_results)
                        pbar.update(len(batch_results))
        
        except Exception as e:
            logger.error(f"Error in parallel simulations: {str(e)}")
            # Fall back to sequential processing
            logger.info("Falling back to sequential processing")
            results = self._sequential_processing(
                state, num_simulations, get_actions
            )
        
        # Calculate batch statistics
        total_time = time.time() - start_time
        stats = self._calculate_statistics(results, total_time)
        
        return results, stats

    def _sequential_processing(
        self,
        state: ISRSWorldState,
        num_simulations: int,
        get_actions: Callable[[ISRSWorldState], List[MultimodalIPPAction]]
    ) -> List[SimulationResult]:
        """Process simulations sequentially as fallback.
        
        Args:
            state: Initial state
            num_simulations: Number of simulations
            get_actions: Function to get available actions
            
        Returns:
            List of simulation results
        """
        seeds = [self.rng.randint(0, 2**32) for _ in range(num_simulations)]
        results = []
        
        for seed in tqdm(seeds, disable=not self.config.show_progress):
            result = _simulate_worker(seed, state)
            if result is not None:
                results.append(result)
                
        return results

    def _calculate_statistics(
        self,
        results: List[SimulationResult],
        total_time: float
    ) -> BatchStatistics:
        """Calculate statistics for a batch of simulations.
        
        Args:
            results: List of simulation results
            total_time: Total computation time
            
        Returns:
            Batch statistics
        """
        if not results:
            return BatchStatistics(0.0, 0.0, 0.0, total_time, 0)
            
        values = [r.value for r in results]
        depths = [r.depth for r in results]
        
        return BatchStatistics(
            mean_value=float(np.mean(values)),
            std_value=float(np.std(values)),
            mean_depth=float(np.mean(depths)),
            total_time=total_time,
            num_simulations=len(results)
        ) 