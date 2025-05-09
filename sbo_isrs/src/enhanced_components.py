"""Enhanced components for POMCP policies including observation widening, belief state clustering, and GP integration."""

from typing import List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .states import ISRSWorldState, ISRSObservation

@dataclass
class ObservationWidener:
    """Widen observations to handle uncertainty.
    
    Attributes:
        width: Width parameter for observation widening
        similarity_threshold: Threshold for observation similarity
        max_width: Maximum width for widening
        min_width: Minimum width for widening
        width_decay: Decay rate for width
    """
    def __init__(
        self,
        width: float = 0.1,
        similarity_threshold: float = 0.6,
        max_width: float = 0.5,
        min_width: float = 0.01,
        width_decay: float = 0.95
    ) -> None:
        """Initialize observation widener.
        
        Args:
            width: Initial width parameter
            similarity_threshold: Threshold for observation similarity
            max_width: Maximum width for widening
            min_width: Minimum width for widening
            width_decay: Decay rate for width
        """
        self.width = width
        self.similarity_threshold = similarity_threshold
        self.max_width = max_width
        self.min_width = min_width
        self.width_decay = width_decay
        self.observations: List[ISRSObservation] = []

    def widen_observation(
        self,
        observation: ISRSObservation
    ) -> List[ISRSObservation]:
        """Widen observation to handle uncertainty.
        
        Args:
            observation: Original observation
            
        Returns:
            List of widened observations
        """
        if not self.observations:
            self.observations.append(observation)
            return [observation]
            
        # Find similar observations
        similar_obs = []
        for obs in self.observations:
            if self._similarity(observation, obs) >= self.similarity_threshold:
                similar_obs.append(obs)
                
        if not similar_obs:
            self.observations.append(observation)
            return [observation]
            
        # Create widened observations
        widened_obs = []
        for obs in similar_obs:
            # Widen reward
            reward_range = self.width * abs(observation.reward - obs.reward)
            min_reward = min(observation.reward, obs.reward) - reward_range
            max_reward = max(observation.reward, obs.reward) + reward_range
            
            # Widen info gain
            info_range = self.width * abs(observation.info_gain - obs.info_gain)
            min_info = max(0, min(observation.info_gain, obs.info_gain) - info_range)
            max_info = max(observation.info_gain, obs.info_gain) + info_range
            
            # Create widened observation
            widened_obs.append(ISRSObservation(
                reward=np.random.uniform(min_reward, max_reward),
                info_gain=np.random.uniform(min_info, max_info),
                location=observation.location
            ))
            
        # Update width
        self.width = max(self.min_width, min(self.max_width, self.width * self.width_decay))
        
        return widened_obs

    def _similarity(
        self,
        obs1: ISRSObservation,
        obs2: ISRSObservation
    ) -> float:
        """Calculate similarity between observations.
        
        Args:
            obs1: First observation
            obs2: Second observation
            
        Returns:
            Similarity score between 0 and 1
        """
        # Location similarity
        loc_sim = 1.0 if obs1.location == obs2.location else 0.0
        
        # Reward similarity
        reward_diff = abs(obs1.reward - obs2.reward)
        reward_sim = np.exp(-reward_diff / 10.0)
        
        # Info gain similarity
        info_diff = abs(obs1.info_gain - obs2.info_gain)
        info_sim = np.exp(-info_diff / 5.0)
        
        # Combine similarities
        return 0.4 * loc_sim + 0.3 * reward_sim + 0.3 * info_sim

class BeliefStateCluster:
    """Cluster of belief states for efficient belief updates.
    
    Attributes:
        states: List of states in cluster
        observations: List of observations in cluster
        gp: GP for belief prediction
        similarity_threshold: Threshold for state similarity
        max_size: Maximum cluster size
    """
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_size: int = 100,
        kernel: Optional[Callable] = None
    ) -> None:
        """Initialize belief state cluster.
        
        Args:
            similarity_threshold: Threshold for state similarity
            max_size: Maximum cluster size
            kernel: GP kernel function
        """
        self.states: List[ISRSWorldState] = []
        self.observations: List[ISRSObservation] = []
        self.gp = GaussianProcessBelief(kernel=kernel)
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size

    def add_state(
        self,
        state: ISRSWorldState,
        observation: ISRSObservation
    ) -> None:
        """Add state and observation to cluster.
        
        Args:
            state: World state
            observation: Observation
        """
        if len(self.states) >= self.max_size:
            # Remove oldest state if cluster is full
            self.states.pop(0)
            self.observations.pop(0)
            
        self.states.append(state)
        self.observations.append(observation)
        
        # Update GP with new state
        x = self.gp.get_feature_vector(state)
        y = self._get_reward(state, observation)
        self.gp.update(x, y)

    def get_similar_states(
        self,
        state: ISRSWorldState
    ) -> List[Tuple[ISRSWorldState, ISRSObservation]]:
        """Get similar states from cluster.
        
        Args:
            state: Query state
            
        Returns:
            List of (state, observation) pairs
        """
        if not self.states:
            return []
            
        # Get GP prediction for query state
        x = self.gp.get_feature_vector(state)
        mean, var = self.gp.predict(x)
        
        # Find similar states based on GP predictions
        similar_states = []
        for s, o in zip(self.states, self.observations):
            s_x = self.gp.get_feature_vector(s)
            s_mean, s_var = self.gp.predict(s_x)
            
            # Compute similarity using GP predictions
            similarity = np.exp(-0.5 * ((mean - s_mean)**2 / (var + s_var)))
            
            if similarity >= self.similarity_threshold:
                similar_states.append((s, o))
                
        return similar_states

    def _get_reward(
        self,
        state: ISRSWorldState,
        observation: ISRSObservation
    ) -> float:
        """Compute reward for state-observation pair.
        
        Args:
            state: World state
            observation: Observation
            
        Returns:
            Reward value
        """
        # Base reward from observation
        reward = observation.reward
        
        # Add information gain
        if observation.info_gain > 0:
            reward += observation.info_gain
            
        # Add exploration bonus
        if state.current not in state.visited:
            reward += 5.0
            
        # Add progress bonus
        progress = len(state.visited) / len(state.location_states)
        reward += 10.0 * progress
        
        return reward

class GaussianProcessBelief:
    """Gaussian Process for belief state representation.
    
    Attributes:
        kernel: GP kernel function
        noise_level: Noise level for GP
        n_restarts: Number of restarts for GP optimization
        X: Feature vectors
        y: Target values
        gp: GP regressor
    """
    def __init__(
        self,
        kernel: Optional[Callable] = None,
        noise_level: float = 1e-4,
        n_restarts: int = 10
    ) -> None:
        """Initialize GP belief.
        
        Args:
            kernel: GP kernel function
            noise_level: Noise level for GP
            n_restarts: Number of restarts for GP optimization
        """
        self.noise_level = noise_level
        self.n_restarts = n_restarts
        self.X = []
        self.y = []
        
        # Use Matern kernel if none provided
        if kernel is None:
            kernel = Matern(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5
            )
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_level,
            n_restarts_optimizer=n_restarts,
            random_state=42
        )

    def update(self, x: np.ndarray, y: float) -> None:
        """Update GP with new observation.
        
        Args:
            x: Feature vector
            y: Target value
        """
        self.X.append(x)
        self.y.append(y)
        
        if len(self.X) > 1:
            X = np.array(self.X)
            y = np.array(self.y)
            self.gp.fit(X, y)

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict mean and variance for given input.
        
        Args:
            x: Feature vector
            
        Returns:
            Mean and variance predictions
        """
        if len(self.X) < 2:
            return 0.0, 1.0
            
        x = x.reshape(1, -1)
        mean, std = self.gp.predict(x, return_std=True)
        return mean[0], std[0]**2

    def get_feature_vector(self, state: ISRSWorldState) -> np.ndarray:
        """Extract feature vector from state.
        
        Args:
            state: World state
            
        Returns:
            Feature vector
        """
        # Location features
        loc_features = np.zeros(len(state.location_states))
        for i, loc_state in enumerate(state.location_states):
            if loc_state == ISRS_STATE.RSGOOD:
                loc_features[i] = 1.0
            elif loc_state == ISRS_STATE.RSBAD:
                loc_features[i] = -1.0
            else:
                loc_features[i] = 0.0
                
        # Visited features
        visited_features = np.zeros(len(state.location_states))
        for loc in state.visited:
            visited_features[loc] = 1.0
            
        # Distance features
        dist_features = np.zeros(len(state.location_states))
        for i in range(len(state.location_states)):
            if i in state.visited:
                dist_features[i] = 0.0
            else:
                dist_features[i] = 1.0 / (1.0 + abs(i - state.current))
                
        # Combine features
        features = np.concatenate([
            loc_features,
            visited_features,
            dist_features,
            [state.cost_expended / 100.0]  # Normalize cost
        ])
        
        return features

    def get_belief(self, state: ISRSWorldState) -> float:
        """Get belief value for state.
        
        Args:
            state: World state
            
        Returns:
            Belief value
        """
        x = self.get_feature_vector(state)
        mean, var = self.predict(x)
        return mean / (1.0 + var)  # Normalize by uncertainty 