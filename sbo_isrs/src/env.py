"""Environment for ISRS simulations."""

from typing import List, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# Local imports
from .actions import MultimodalIPPAction, ISRSSensor
from .states import ISRSWorldState, ISRSObservation, ISRS_STATE
from .belief import ISRSBelief
from .belief_types import ISRSLocationBelief

@dataclass
class GPFeatures:
    """Features for GP regression.
    
    Attributes:
        location_coords: Location coordinates
        location_states: List of location states
        visited: Set of visited locations
        current: Current location
    """
    location_coords: np.ndarray
    location_states: List[ISRS_STATE]
    visited: Set[int]
    current: int
    
    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for GP regression.
        
        Returns:
            Feature vector
        """
        # Compute distances to all locations
        current_coord = self.location_coords[self.current]
        distances = np.linalg.norm(self.location_coords - current_coord, axis=1)
        
        # Compute state indicators
        good_indicators = np.array([s == ISRS_STATE.RSGOOD for s in self.location_states])
        bad_indicators = np.array([s == ISRS_STATE.RSBAD for s in self.location_states])
        beacon_indicators = np.array([s == ISRS_STATE.RSBEACON for s in self.location_states])
        
        # Compute visit indicators
        visit_indicators = np.array([i in self.visited for i in range(len(self.location_states))])
        
        # Combine features
        features = np.concatenate([
            distances,
            good_indicators,
            bad_indicators,
            beacon_indicators,
            visit_indicators
        ])
        
        return features
        
    def get_location_features(self, location: int) -> np.ndarray:
        """Get features for a specific location.
        
        Args:
            location: Location index
            
        Returns:
            Feature vector for the location
        """
        # Compute distance to target location
        target_coord = self.location_coords[location]
        distances = np.linalg.norm(self.location_coords - target_coord, axis=1)
        
        # Compute state indicators for target location
        good_indicator = float(self.location_states[location] == ISRS_STATE.RSGOOD)
        bad_indicator = float(self.location_states[location] == ISRS_STATE.RSBAD)
        beacon_indicator = float(self.location_states[location] == ISRS_STATE.RSBEACON)
        
        # Compute visit indicator for target location
        visit_indicator = float(location in self.visited)
        
        # Combine features
        features = np.array([
            distances[location],
            good_indicator,
            bad_indicator,
            beacon_indicator,
            visit_indicator
        ])
        
        return features
        
    @property
    def uncertainty(self) -> float:
        """Get uncertainty value for current location.
        
        Returns:
            Uncertainty value between 0 and 1
        """
        # Higher uncertainty for unvisited locations
        if self.current not in self.visited:
            return 1.0
            
        # Lower uncertainty for visited locations based on state
        state = self.location_states[self.current]
        if state == ISRS_STATE.UNKNOWN:
            return 1.0
        elif state == ISRS_STATE.RSGOOD or state == ISRS_STATE.RSBAD:
            return 0.0
        else:  # RSBEACON or RSNEITHER
            return 0.5

class ISRSEnv:
    """Environment for ISRS simulations."""
    
    def __init__(
        self,
        num_locations: int,
        num_good: int,
        num_bad: int,
        num_beacons: int,
        sensor_efficiency: float,
        seed: Optional[int] = None
    ) -> None:
        """Initialize environment.
        
        Args:
            num_locations: Number of locations
            num_good: Number of good samples
            num_bad: Number of bad samples
            num_beacons: Number of beacons
            sensor_efficiency: Sensor efficiency
            seed: Random seed
        """
        self.num_locations = num_locations
        self.n_locations = num_locations  # Alias for compatibility
        self.num_good = num_good
        self.num_bad = num_bad
        self.num_beacons = num_beacons
        self.sensor_efficiency = sensor_efficiency
        self.sensing_efficiencies = [0.6, 0.8, 1.0]  # LOW, MEDIUM, HIGH
        self.rng = np.random.RandomState(seed)
        
        # Initialize location metadata (coordinates in a grid)
        grid_size = int(np.ceil(np.sqrt(num_locations)))
        self.location_metadata = []
        for i in range(num_locations):
            x = i // grid_size
            y = i % grid_size
            self.location_metadata.append((x, y))
            
        # Calculate shortest paths between all locations
        self.shortest_paths = self._calculate_shortest_paths()
        
        # Initialize state
        self.state = None
        self.observation = None
        self.belief = None
        
        # Initialize GP features
        self.gp_features = None
        
    def update_gp_features(self, state: ISRSWorldState) -> None:
        """Update GP features based on current state.
        
        Args:
            state: Current world state
        """
        # Create feature vector for GP regression
        location_coords = np.array([
            [i // int(np.sqrt(self.num_locations)), i % int(np.sqrt(self.num_locations))]
            for i in range(self.num_locations)
        ])
        
        # Update GP features
        self.gp_features = GPFeatures(
            location_coords=location_coords,
            location_states=state.location_states,
            visited=state.visited,
            current=state.current
        )
        
    def _calculate_shortest_paths(self) -> np.ndarray:
        """Calculate shortest paths between all locations.
        
        Returns:
            Matrix of shortest paths between locations
        """
        # Calculate Euclidean distances between all locations
        distances = np.zeros((self.num_locations, self.num_locations))
        for i in range(self.num_locations):
            for j in range(i + 1, self.num_locations):
                dist = np.linalg.norm(
                    np.array(self.location_metadata[i]) - 
                    np.array(self.location_metadata[j])
                )
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
        
    def _initialize_world_state(self) -> ISRSWorldState:
        """Initialize world state with random rock samples."""
        # Create list of all possible states
        states = [ISRS_STATE.RSGOOD] * self.num_good
        states.extend([ISRS_STATE.RSBAD] * self.num_bad)
        states.extend([ISRS_STATE.RSBEACON] * self.num_beacons)
        states.extend([ISRS_STATE.RSNEITHER] * (self.num_locations - len(states)))
        
        # Shuffle states
        self.rng.shuffle(states)
        
        return ISRSWorldState(
            current=0,
            visited={0},
            location_states=states,
            cost_expended=0.0
        )
        
    def _initialize_belief_state(self) -> ISRSBelief:
        """Initialize belief state with uniform distribution."""
        # Initialize uniform beliefs for each location
        location_beliefs = []
        for _ in range(self.num_locations):
            # Uniform distribution over RSGOOD, RSBAD, RSBEACON, RSNEITHER
            # We don't include UNKNOWN in the belief state
            probs = np.ones(4) / 4  
            location_beliefs.append(ISRSLocationBelief(probs))
            
        return ISRSBelief(
            current=0,
            visited={0},
            location_beliefs=location_beliefs,
            cost_expended=0.0
        )
        
    def reset(self) -> Tuple[ISRSWorldState, ISRSObservation, ISRSBelief]:
        """Reset environment to initial state.
        
        Returns:
            Tuple of (initial state, initial observation, initial belief)
        """
        # Initialize world state
        self.state = self._initialize_world_state()
        
        # Get initial observation (no action taken yet)
        self.observation = ISRSObservation(
            current=self.state.current,
            visited=self.state.visited,
            location_states=[ISRS_STATE.UNKNOWN] * self.num_locations,
            cost_expended=0.0
        )
        
        # Initialize belief state
        self.belief = self._initialize_belief_state()
        
        return self.state, self.observation, self.belief
        
    def step(
        self,
        action: MultimodalIPPAction
    ) -> Tuple[ISRSWorldState, ISRSObservation, float, bool, ISRSBelief]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next state, observation, reward, done, belief)
        """
        # Update state based on action
        next_state = self._update_world_state(action)
        
        # Get observation
        observation = self._get_observation(next_state, action)
        
        # Calculate reward
        reward = self._calculate_reward(next_state, action)
        
        # Check if episode is done
        done = self._is_done(next_state)
        
        # Update belief
        belief = self._update_belief(next_state, observation, action)
        
        # Update current state
        self.state = next_state
        
        return next_state, observation, reward, done, belief
        
    def _update_world_state(self, action: MultimodalIPPAction) -> ISRSWorldState:
        """Update world state based on action.
        
        Args:
            action: Action to take
            
        Returns:
            Updated world state
        """
        # Create copy of current state
        next_state = ISRSWorldState(
            current=self.state.current,
            visited=self.state.visited.copy(),
            location_states=self.state.location_states.copy(),
            cost_expended=self.state.cost_expended
        )
        
        # Update state based on action type
        if action.visit_location is not None:
            # Update current location
            next_state.current = action.visit_location
            next_state.visited.add(action.visit_location)
            next_state.cost_expended += action.cost
        elif action.sensing_action is not None:
            # Update sensing cost
            next_state.cost_expended += action.cost
            
            # Update location state if sensing is successful
            if self.rng.random() < action.sensing_action.value:
                next_state.location_states[next_state.current] = self.state.location_states[next_state.current]
        
        return next_state
            
    def _get_observation(self, state: ISRSWorldState, action: MultimodalIPPAction) -> ISRSObservation:
        """Get observation for current state and action.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Observation
        """
        # Initialize observation with unknown states
        location_states = [ISRS_STATE.UNKNOWN] * self.num_locations
        
        # Update observation based on action type
        if action.visit_location is not None:
            # Perfect observation at target location
            location_states[action.visit_location] = state.location_states[action.visit_location]
        elif action.sensing_action is not None:
            # Noisy observation based on sensor efficiency
            if self.rng.random() < action.sensing_action.value:
                location_states[state.current] = state.location_states[state.current]
        
        return ISRSObservation(
            current=state.current,
            visited=state.visited,
            location_states=location_states,
            cost_expended=state.cost_expended
        )
        
    def _is_done(self, state: ISRSWorldState) -> bool:
        """Check if episode is done.
        
        Returns:
            True if episode is done
        """
        return len(state.visited) == self.num_locations
        
    def observation(
        self,
        current: int,
        sensor: ISRSSensor,
        states: List[int],
        rng: np.random.RandomState
    ) -> List[int]:
        """Generate observation given current state and sensor.
        
        Args:
            current: Current location
            sensor: Sensor used
            states: Current states
            rng: Random number generator
            
        Returns:
            List of observations
        """
        obs = states.copy()
        for i in range(len(obs)):
            if rng.random() > sensor.efficiency:
                obs[i] = ISRS_STATE.UNKNOWN
        return obs 

    def _initialize_location_metadata(self) -> List[Tuple[float, float]]:
        """Initialize location metadata.
        
        Returns:
            List of (x, y) coordinates for each location
        """
        # Generate random coordinates in [0, 1] x [0, 1]
        coords = []
        for _ in range(self.num_locations):
            x = self.rng.uniform(0, 1)
            y = self.rng.uniform(0, 1)
            coords.append((x, y))
        return coords 

    def _calculate_reward(self, state: ISRSWorldState, action: MultimodalIPPAction) -> float:
        """Calculate reward for current state and action.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Calculate reward based on action type
        if action.visit_location is not None:
            # Reward for visiting good locations
            if state.location_states[action.visit_location] == ISRS_STATE.RSGOOD:
                reward += 15.0
            elif state.location_states[action.visit_location] == ISRS_STATE.RSBAD:
                reward -= 3.0
            elif state.location_states[action.visit_location] == ISRS_STATE.RSBEACON:
                reward += 2.0
            
            # Cost for movement
            reward -= action.cost
        elif action.sensing_action is not None:
            # Cost for sensing
            reward -= action.cost
        
        return reward

    def _update_belief(
        self,
        state: ISRSWorldState,
        observation: ISRSObservation,
        action: MultimodalIPPAction
    ) -> ISRSBelief:
        """Update belief based on state, observation, and action.
        
        Args:
            state: Current state
            observation: Current observation
            action: Action taken
            
        Returns:
            Updated belief
        """
        # Create initial belief if none exists
        if not hasattr(self, 'belief'):
            self.belief = ISRSBelief(
                current=state.current,
                visited=state.visited,
                location_beliefs=[ISRSLocationBelief() for _ in range(self.num_locations)],
                cost_expended=state.cost_expended
            )
            
        # Update belief
        return self.belief.update(action, observation, state) 