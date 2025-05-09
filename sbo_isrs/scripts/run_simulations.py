"""Script to run ISRS environment simulations."""

import os
import sys
import argparse
from typing import List, Dict
import numpy as np
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import ISRSEnv
from src.pomdp import ISRSPOMDP
from src.policies import (
    RandomPolicy,
    GreedyPolicy,
    POMCPPolicy,
    InformationSeekingPolicy,
    get_pomcp_dpw_policy
)
from src.simulator import run_simulation, SimulationResult
from src.reward_calculator import RewardCalculator

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run ISRS simulations')
    parser.add_argument('--num_locations', type=int, default=10, help='Number of locations')
    parser.add_argument('--num_good', type=int, default=3, help='Number of good samples')
    parser.add_argument('--num_bad', type=int, default=3, help='Number of bad samples')
    parser.add_argument('--num_beacons', type=int, default=2, help='Number of beacons')
    parser.add_argument('--sensor_efficiency', type=float, default=0.8, help='Sensor efficiency')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--policy_types',
        nargs='+',
        default=['random', 'dpw'],
        help='Policy types to evaluate'
    )
    return parser.parse_args()

def analyze_results(results: List[SimulationResult]) -> Dict[str, float]:
    """Analyze simulation results.
    
    Args:
        results: List of simulation results
        
    Returns:
        Dictionary of statistics
    """
    total_rewards = [r.total_reward for r in results]
    steps_taken = [r.steps_taken for r in results]
    belief_errors = [np.mean(r.belief_errors) for r in results]
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_steps': np.mean(steps_taken),
        'std_steps': np.std(steps_taken),
        'mean_belief_error': np.mean(belief_errors),
        'std_belief_error': np.std(belief_errors)
    }

def main() -> None:
    """Run simulations with different policies."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ISRS simulations')
    parser.add_argument('--num_locations', type=int, default=10, help='Number of locations')
    parser.add_argument('--num_good', type=int, default=3, help='Number of good samples')
    parser.add_argument('--num_bad', type=int, default=3, help='Number of bad samples')
    parser.add_argument('--num_beacons', type=int, default=2, help='Number of beacons')
    parser.add_argument('--sensor_efficiency', type=float, default=0.8, help='Sensor efficiency')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--policy_types',
        nargs='+',
        default=['random', 'dpw'],
        help='Policy types to evaluate'
    )
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    env = ISRSEnv(
        num_locations=args.num_locations,
        num_good=args.num_good,
        num_bad=args.num_bad,
        num_beacons=args.num_beacons,
        sensor_efficiency=args.sensor_efficiency,
        seed=args.seed
    )
    
    # Create POMDP model
    pomdp = ISRSPOMDP(env)
    
    # Run simulations for each policy type
    for policy_type in args.policy_types:
        if policy_type not in ['random', 'dpw']:
            logger.warning(f"Skipping unsupported policy type: {policy_type}")
            continue
        logger.info(f"Running simulations with {policy_type} policy")
        
        # Create policy
        rng = np.random.RandomState(args.seed)
        if policy_type == 'random':
            policy = RandomPolicy(pomdp=pomdp, rng=rng)
        elif policy_type == 'greedy':
            policy = GreedyPolicy(pomdp=pomdp, rng=rng)
        elif policy_type == 'pomcp':
            policy = POMCPPolicy(pomdp=pomdp, rng=rng)
        elif policy_type == 'information':
            policy = InformationSeekingPolicy(pomdp=pomdp, rng=rng)
        elif policy_type == 'dpw':
            policy = get_pomcp_dpw_policy(pomdp=pomdp, rng=rng)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Run simulation
        results = run_simulation(
            env=env,
            policy=policy,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed
        )
        
        # Analyze results
        stats = analyze_results(results)
        
        # Log results
        logger.info(f"Results for {policy_type} policy:")
        logger.info(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        logger.info(f"  Mean steps: {stats['mean_steps']:.1f} ± {stats['std_steps']:.1f}")
        logger.info(f"  Mean belief error: {stats['mean_belief_error']:.4f} ± {stats['std_belief_error']:.4f}")

if __name__ == '__main__':
    main() 