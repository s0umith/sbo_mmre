"""Plotting utilities for the Rover environment."""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ..core.beliefs import RoverLocationBelief
import os

def plot_trial(
    true_map: np.ndarray,
    state_hist: List[Tuple[int, int]],
    belief_hist: List[RoverLocationBelief],
    action_hist: List[np.ndarray],
    total_reward_hist: List[float],
    reward_hist: List[float],
    trial_num: int,
    policy_name: str,
    save_dir: str = "figures"
) -> None:
    """Plot trial results.
    
    Args:
        true_map: True environment map
        state_hist: History of states
        belief_hist: History of beliefs
        action_hist: History of actions
        total_reward_hist: History of cumulative rewards
        reward_hist: History of individual rewards
        trial_num: Trial number
        policy_name: Name of policy used
        save_dir: Directory to save plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot true map
    im1 = ax1.imshow(true_map, cmap='viridis')
    ax1.set_title('True Map')
    plt.colorbar(im1, ax=ax1)
    
    # Plot trajectory on true map
    for i in range(len(state_hist)-1):
        start = state_hist[i]
        end = state_hist[i+1]
        ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', alpha=0.5)
    ax1.plot([x for x, _ in state_hist], [y for _, y in state_hist], 'r.', label='Trajectory')
    ax1.legend()
    
    # Plot final belief mean
    final_belief = belief_hist[-1].get_belief_map()
    im2 = ax2.imshow(final_belief, cmap='viridis')
    ax2.set_title('Final Belief Mean')
    plt.colorbar(im2, ax=ax2)
    
    # Plot final uncertainty
    final_uncertainty = belief_hist[-1].get_uncertainty_map()
    im3 = ax3.imshow(final_uncertainty, cmap='viridis')
    ax3.set_title('Final Uncertainty')
    plt.colorbar(im3, ax=ax3)
    
    # Plot rewards
    ax4.plot(reward_hist, label='Step Reward')
    ax4.plot(total_reward_hist, label='Cumulative Reward')
    ax4.set_title('Rewards')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Reward')
    ax4.legend()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'trial_{trial_num}_{policy_name}.png'))
    plt.close()

def plot_rmse_history(
    rmse_hist: List[List[float]],
    policy_name: str,
    save_dir: str = "figures"
) -> None:
    """Plot RMSE history across trials.
    
    Args:
        rmse_hist: History of RMSE values for each trial
        policy_name: Name of policy used
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual trial RMSE
    for trial_rmse in rmse_hist:
        plt.plot(trial_rmse, alpha=0.3)
    
    # Plot mean RMSE
    mean_rmse = np.mean(rmse_hist, axis=0)
    plt.plot(mean_rmse, 'k-', linewidth=2, label='Mean RMSE')
    
    plt.title(f'RMSE History - {policy_name}')
    plt.xlabel('Step')
    plt.ylabel('RMSE')
    plt.legend()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'rmse_history_{policy_name}.png'))
    plt.close()

def plot_trace_history(
    trace_hist: List[List[float]],
    policy_name: str,
    save_dir: str = "figures"
) -> None:
    """Plot trace history across trials.
    
    Args:
        trace_hist: History of trace values for each trial
        policy_name: Name of policy used
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual trial traces
    for trial_trace in trace_hist:
        plt.plot(trial_trace, alpha=0.3)
    
    # Plot mean trace
    mean_trace = np.mean(trace_hist, axis=0)
    plt.plot(mean_trace, 'k-', linewidth=2, label='Mean Trace')
    
    plt.title(f'Trace History - {policy_name}')
    plt.xlabel('Step')
    plt.ylabel('Trace')
    plt.legend()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'trace_history_{policy_name}.png'))
    plt.close()

def calculate_rmse(
    true_map: np.ndarray,
    belief: RoverLocationBelief
) -> float:
    """Calculate RMSE between true map and belief.
    
    Args:
        true_map: True environment map
        belief: Current belief state
        
    Returns:
        RMSE value
    """
    belief_map = belief.get_belief_map()
    return np.sqrt(np.mean((true_map - belief_map) ** 2))

def calculate_trace(belief: RoverLocationBelief) -> float:
    """Calculate trace of belief covariance.
    
    Args:
        belief: Current belief state
        
    Returns:
        Trace value
    """
    uncertainty_map = belief.get_uncertainty_map()
    return np.sum(uncertainty_map) 