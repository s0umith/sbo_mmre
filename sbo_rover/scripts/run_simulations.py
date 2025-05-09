"""Script to run rover environment simulations and compare different policies."""

import os
import sys
import argparse
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rover_sbo.env.rover_env import create_rover_env
from src.rover_sbo.core.actions import RoverAction, DIRECTIONS
from src.rover_sbo.policies.basic import BasicPolicy
from src.rover_sbo.policies.enhanced_gp_mcts import EnhancedGPMCTSPolicy
from src.rover_sbo.policies.pomcp import POMCPPolicy
from src.rover_sbo.policies.raster import RasterPolicy

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run rover simulations')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10], help='Grid size (rows, cols)')
    parser.add_argument('--num_sample_types', type=int, default=3, help='Number of sample types')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--policy_types',
        nargs='+',
        default=['pomcp', 'basic', 'enhanced', 'raster'],
        help='Policy types to evaluate'
    )
    return parser.parse_args()

def analyze_results(results: List[Dict]) -> Dict[str, float]:
    """Analyze simulation results.
    
    Args:
        results: List of simulation results
        
    Returns:
        Dictionary of statistics
    """
    total_rewards = [r['total_reward'] for r in results]
    steps_taken = [r['steps_taken'] for r in results]
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_steps': np.mean(steps_taken),
        'std_steps': np.std(steps_taken)
    }

def action_to_vector(action: RoverAction) -> np.ndarray:
    """Convert RoverAction to action vector.
    
    Args:
        action: RoverAction object
        
    Returns:
        Action vector (dx, dy)
    """
    return np.array(DIRECTIONS[action.action_type])

def plot_results(results: Dict[str, Dict[str, float]], save_path: str = "policy_comparison.png") -> None:
    """Plot comparison of policy results.
    
    Args:
        results: Dictionary of results for each policy
        save_path: Path to save the plot
    """
    policies = list(results.keys())
    rewards = [results[p]['mean_reward'] for p in policies]
    reward_stds = [results[p]['std_reward'] for p in policies]
    steps = [results[p]['mean_steps'] for p in policies]
    step_stds = [results[p]['std_steps'] for p in policies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot rewards
    ax1.bar(policies, rewards, yerr=reward_stds, capsize=5)
    ax1.set_title('Average Rewards')
    ax1.set_ylabel('Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot steps
    ax2.bar(policies, steps, yerr=step_stds, capsize=5)
    ax2.set_title('Average Steps')
    ax2.set_ylabel('Steps')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main() -> None:
    """Run simulations with different policies."""
    args = parse_args()
    
    # Set random seed
    rng = np.random.RandomState(args.seed)
    
    # Create environment
    env = create_rover_env(
        grid_size=tuple(args.grid_size),
        num_sample_types=args.num_sample_types,
        seed=args.seed
    )
    
    # Initialize results dictionary
    all_results = {}
    
    # Run simulations for each policy type
    for policy_type in args.policy_types:
        logger.info(f"Running simulations with {policy_type} policy")
        
        # Create policy
        if policy_type == 'basic':
            policy = BasicPolicy(env=env, rng=rng, exploration_prob=0.1)
        elif policy_type == 'enhanced':
            policy = EnhancedGPMCTSPolicy(env=env)
        elif policy_type == 'pomcp':
            policy = POMCPPolicy(env=env)
        elif policy_type == 'raster':
            policy = RasterPolicy(grid_size=env.grid_size, drill_interval=3)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Run simulation
        results = []
        for episode in range(args.num_episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < args.max_steps:
                if policy_type == 'raster':
                    action = policy.get_action(state)
                    next_state, reward, done, info = env.step(action)
                else:
                    rover_action = policy.get_action(env.belief)
                    action = action_to_vector(rover_action)
                    next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
                steps += 1
            
            results.append({
                'total_reward': total_reward,
                'steps_taken': steps
            })
        
        # Analyze results
        stats = analyze_results(results)
        all_results[policy_type] = stats
        
        # Log results
        logger.info(f"Results for {policy_type} policy:")
        logger.info(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    
    # Plot results
    plot_results(all_results)

if __name__ == '__main__':
    main()