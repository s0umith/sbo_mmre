"""Belief state representations for the Rover environment."""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class RoverLocationBelief:
    """Belief state about rover's location using Gaussian Process."""
    
    def __init__(self, grid_size: Tuple[int, int], kernel_params: Optional[Dict] = None):
        """Initialize belief state.
        
        Args:
            grid_size: Size of the grid (width, height)
            kernel_params: Parameters for the GP kernel
        """
        self.grid_size = grid_size
        self.width, self.height = grid_size
        
        # Initialize GP with fixed parameters to avoid convergence warnings
        kernel = RBF(length_scale=0.5, length_scale_bounds="fixed") + \
                 WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-2,  # Increased regularization
            normalize_y=True,
            n_restarts_optimizer=0  # Disable optimization since parameters are fixed
        )
        
        # Initialize training data
        self.X = np.array([]).reshape(0, 2)  # Positions
        self.y = np.array([])  # Values
        
    def update(self, state: np.ndarray, action: np.ndarray) -> None:
        """Update belief with new observation.
        
        Args:
            state: State vector containing position and observation
            action: Action vector [dx, dy]
        """
        # Extract position from state
        pos_x, pos_y = state[:2]
        position = np.array([pos_x, pos_y])
        
        # Extract observation from state
        grid_size = self.width * self.height
        belief_map = state[2:2+grid_size].reshape(self.height, self.width)
        value = belief_map[int(pos_y), int(pos_x)]
        
        # Add new observation to training data
        self.X = np.vstack([self.X, position])
        self.y = np.append(self.y, value)
        
        # Refit GP
        self.gp.fit(self.X, self.y)
        
    def predict(self, position: Tuple[int, int]) -> Tuple[float, float]:
        """Predict value and uncertainty at given position.
        
        Args:
            position: Position to predict (x, y)
            
        Returns:
            Tuple of (mean, std_dev) at position
        """
        if len(self.X) == 0:
            return 0.0, 1.0
            
        mean, std = self.gp.predict(
            np.array(position).reshape(1, -1),
            return_std=True
        )
        return mean[0], std[0]
        
    def get_belief_map(self) -> np.ndarray:
        """Get full belief map over grid.
        
        Returns:
            2D array of mean predictions
        """
        # Create grid of positions
        x = np.arange(self.width)
        y = np.arange(self.height)
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Predict values
        if len(self.X) == 0:
            return np.zeros((self.height, self.width))
            
        means = self.gp.predict(positions)
        return means.reshape(self.height, self.width)
        
    def get_uncertainty_map(self) -> np.ndarray:
        """Get uncertainty map over grid.
        
        Returns:
            2D array of standard deviations
        """
        # Create grid of positions
        x = np.arange(self.width)
        y = np.arange(self.height)
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Predict uncertainties
        if len(self.X) == 0:
            return np.ones((self.height, self.width))
            
        _, stds = self.gp.predict(positions, return_std=True)
        return stds.reshape(self.height, self.width) 