"""SBO++ package for Information Seeking Rock Sample (ISRS) environment (sbo_isrs, part of pp-sbo)."""

from typing import TYPE_CHECKING

from .src import actions
from .src import states
from .src import belief
from .src import belief_types
from .src import belief_updates
from .src import env
from .src import policies
from .src import pomdp
from .src import pomcp
from .src import reward_calculator
from .src import reward_components
from .src import simulator
from .src import parallel_simulator
from .src import trials
from .src import enhanced_components
from .src import enhanced_observations

if TYPE_CHECKING:
    from .src.actions import MultimodalIPPAction, ISRSSensor
    from .src.states import ISRSWorldState, ISRSObservation, ISRSBelief, ISRS_STATE
    from .src.belief import ISRSBelief
    from .src.belief_types import ISRSLocationBelief
    from .src.env import ISRSEnv
    from .src.policies import (
        BasePolicy, RandomPolicy, GreedyPolicy, POMCPPolicy,
        InformationSeekingPolicy, get_pomcp_dpw_policy
    )
    from .src.pomdp import ISRSPOMDP
    from .src.pomcp import POMCP
    from .src.reward_calculator import RewardCalculator
    from .src.simulator import run_simulation, SimulationResult
    from .src.parallel_simulator import ParallelSimulator, SimulationConfig, RolloutStrategy

__version__ = "0.1.0"

__all__ = [
    "actions",
    "states", 
    "belief",
    "belief_types",
    "belief_updates",
    "env",
    "policies",
    "pomdp",
    "pomcp",
    "reward_calculator",
    "reward_components",
    "simulator",
    "parallel_simulator",
    "trials",
    "enhanced_components",
    "enhanced_observations"
] 