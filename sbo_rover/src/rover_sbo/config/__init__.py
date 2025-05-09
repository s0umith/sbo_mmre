"""Configuration settings for the Rover SBO package (part of pp-sbo)."""

from typing import TYPE_CHECKING, Dict, Any

# Default configuration settings
DEFAULT_CONFIG: Dict[str, Any] = {
    "map_size": (5, 5),
    "num_good_samples": 3,
    "num_bad_samples": 3,
    "num_beacons": 2,
    "good_sample_reward": 1.0,
    "bad_sample_penalty": -1.0,
    "init_pos": (1, 1),
    "cost_budget": 300.0,
    "drill_time": 3.0,
    "step_size": 1,
    "discount_factor": 1.0,
    "sigma_drill": 1e-9,
    "sigma_spec": 0.1,
}

if TYPE_CHECKING:
    from typing import TypedDict

    class ConfigDict(TypedDict):
        map_size: tuple[int, int]
        num_good_samples: int
        num_bad_samples: int
        num_beacons: int
        good_sample_reward: float
        bad_sample_penalty: float
        init_pos: tuple[int, int]
        cost_budget: float
        drill_time: float
        step_size: int
        discount_factor: float
        sigma_drill: float
        sigma_spec: float

__all__ = ["DEFAULT_CONFIG"] 