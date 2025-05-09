"""Module for running ISRS trials."""

from typing import Dict, List, Tuple, Any
import numpy as np
import time
from loguru import logger
from tqdm import tqdm

from .actions import MultimodalIPPAction
from .states import ISRSWorldState, ISRSObservation
from .pomdp import ISRSPOMDP
from .policies import POMCPPolicy
from .env import ISRS_STATE

def run_trial(
    pomdp: ISRSPOMDP,
    policy: POMCPPolicy,
    rng: np.random.RandomState
) -> Tuple[float, List[ISRSWorldState], List[MultimodalIPPAction], List[ISRSObservation], List[float], float, int]:
    """Run a single trial.
    
    Args:
        pomdp: POMDP instance
        policy: Policy to use
        rng: Random number generator
        
    Returns:
        Tuple containing:
        - Total reward
        - State history
        - Action history
        - Observation history
        - Reward history
        - Planning time
        - Number of plans
    """
    # Initialize state and belief
    state = pomdp.sample_initial_state(rng)
    belief = pomdp.initial_belief_state()
    
    # Initialize histories
    state_hist = [state]
    action_hist = []
    obs_hist = []
    reward_hist = []
    
    # Initialize metrics
    total_reward = 0.0
    total_planning_time = 0.0
    num_plans = 0
    
    # Run trial
    while not pomdp.is_terminal(state):
        # Get action
        start_time = time.time()
        action = policy.get_action(belief)
        total_planning_time += time.time() - start_time
        num_plans += 1
        
        # Take action
        next_state = pomdp.generate_s(state, action, rng)
        observation = pomdp.generate_o(state, action, next_state, rng)
        reward = pomdp.get_reward(state, action, next_state)
        total_reward += reward
        
        # Update histories
        state_hist.append(next_state)
        action_hist.append(action)
        obs_hist.append(observation)
        reward_hist.append(reward)
        
        # Update belief
        belief = pomdp.update_belief(belief, action, observation)
        
        # Update state
        state = next_state
        
    return (
        total_reward,
        state_hist,
        action_hist,
        obs_hist,
        reward_hist,
        total_planning_time,
        num_plans
    )

def run_policy_comparison(
    pomdp: ISRSPOMDP,
    policies: Dict[str, POMCPPolicy],
    num_trials: int = 10,
    rng: np.random.RandomState = None,
    log_trace_rmse: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Run policy comparison.
    
    Args:
        pomdp: POMDP instance
        policies: Dictionary of policies to compare
        num_trials: Number of trials to run
        rng: Random number generator
        log_trace_rmse: Whether to log RMSE and trace
        
    Returns:
        Dictionary of results for each policy
    """
    # Initialize results
    results = {}
    
    # Run trials for each policy
    for policy_name, policy in policies.items():
        logger.info(f"Running trials for {policy_name}...")
        
        # Initialize metrics
        total_reward = 0.0
        total_planning_time = 0.0
        total_plans = 0
        rmse_hist = []
        trace_hist = []
        
        # Run trials
        for _ in tqdm(range(num_trials)):
            # Run trial
            reward, state_hist, action_hist, obs_hist, reward_hist, planning_time, num_plans = run_trial(
                pomdp, policy, rng
            )
            
            # Update metrics
            total_reward += reward
            total_planning_time += planning_time
            total_plans += num_plans
            
            # Calculate RMSE and trace if requested
            if log_trace_rmse:
                # Calculate RMSE
                rmse = calculate_rmse_along_traj(
                    pomdp, state_hist, action_hist, obs_hist, reward_hist, reward_hist
                )
                rmse_hist.append(rmse)
                
                # Calculate trace
                trace = calculate_trace_Σ(
                    pomdp, state_hist, action_hist, obs_hist, reward_hist, reward_hist
                )
                trace_hist.append(trace)
            
        # Store results
        results[policy_name] = {
            "avg_reward": total_reward / num_trials,
            "avg_planning_time": total_planning_time / total_plans,
            "avg_rmse": np.mean(rmse_hist) if log_trace_rmse else None,
            "avg_trace": np.mean(trace_hist) if log_trace_rmse else None,
            "rmse_hist": rmse_hist if log_trace_rmse else None,
            "trace_hist": trace_hist if log_trace_rmse else None
        }
        
    return results

def calculate_rmse_along_traj(
    pomdp: ISRSPOMDP,
    state_hist: List[ISRSWorldState],
    action_hist: List[MultimodalIPPAction],
    obs_hist: List[ISRSObservation],
    total_reward_hist: List[float],
    reward_hist: List[float]
) -> float:
    """Calculate RMSE along trajectory.
    
    Args:
        pomdp: POMDP instance
        state_hist: History of states
        action_hist: History of actions
        obs_hist: History of observations
        total_reward_hist: History of total rewards
        reward_hist: History of rewards
        
    Returns:
        RMSE value
    """
    # Get true rock states
    true_states = state_hist[-1].location_states
    
    # Get final belief
    belief = pomdp.initial_belief_state()
    for action, obs in zip(action_hist, obs_hist):
        belief = pomdp.update_belief(belief, action, obs)
    
    # Calculate RMSE
    squared_errors = []
    for i, true_state in enumerate(true_states):
        if true_state in [ISRS_STATE.RSGOOD, ISRS_STATE.RSBAD]:
            est_probs = belief.location_beliefs[i].probs
            true_prob = 1.0 if true_state == ISRS_STATE.RSGOOD else 0.0
            error = true_prob - est_probs[0]  # Probability of being good
            squared_errors.append(error ** 2)
    
    return np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

def calculate_trace_Σ(
    pomdp: ISRSPOMDP,
    state_hist: List[ISRSWorldState],
    action_hist: List[MultimodalIPPAction],
    obs_hist: List[ISRSObservation],
    total_reward_hist: List[float],
    reward_hist: List[float]
) -> float:
    """Calculate trace of belief covariance matrix.
    
    Args:
        pomdp: POMDP instance
        state_hist: History of states
        action_hist: History of actions
        obs_hist: History of observations
        total_reward_hist: History of total rewards
        reward_hist: History of rewards
        
    Returns:
        Trace value
    """
    # Get final belief
    belief = pomdp.initial_belief_state()
    for action, obs in zip(action_hist, obs_hist):
        belief = pomdp.update_belief(belief, action, obs)
    
    # Calculate trace
    _, trace = belief.get_metrics(state_hist[-1])
    return trace
