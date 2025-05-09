"""Gaussian Process implementation for ISRS environment."""

from typing import List, Tuple
import numpy as np
from scipy.linalg import solve_triangular, cholesky
from scipy.spatial.distance import cdist

from .position import Position

class GaussianProcess:
    """Gaussian Process for belief tracking.
    
    Attributes:
        m: Mean function
        mXq: Mean function at query points
        k: Covariance function (squared exponential kernel)
        X: Design points
        X_query: Query points
        y: Objective values
        ν: Noise variance
        KXX: Kernel matrix for design points
        KXqX: Kernel matrix between query and design points
        KXqXq: Kernel matrix for query points
    """
    def __init__(self, lengthscale: float = 0.5):
        """Initialize Gaussian Process.
        
        Args:
            lengthscale: Lengthscale parameter for squared exponential kernel
        """
        self.m = lambda x: 0.5  # Constant mean function
        self.mXq = None
        self.lengthscale = lengthscale
        self.X = []
        self.X_query = []
        self.y = []
        self.ν = []
        self.KXX = None
        self.KXqX = None
        self.KXqXq = None
        
    def _squared_exponential_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute squared exponential kernel matrix.
        
        Args:
            x1: First set of points (n1 x d)
            x2: Second set of points (n2 x d)
            
        Returns:
            Kernel matrix (n1 x n2)
        """
        # Compute squared distances
        dists = cdist(x1, x2, 'sqeuclidean')
        
        # Compute kernel matrix
        return np.exp(-dists / (2 * self.lengthscale**2))
        
    def _kernel_matrix(self, X1: List[Position], X2: List[Position]) -> np.ndarray:
        """Compute kernel matrix between two sets of points.
        
        Args:
            X1: First set of points
            X2: Second set of points
            
        Returns:
            Kernel matrix
        """
        # Convert positions to numpy arrays
        X1_arr = np.array([[x.x, x.y] for x in X1])
        X2_arr = np.array([[x.x, x.y] for x in X2])
        
        # Compute kernel matrix using squared exponential kernel
        return self._squared_exponential_kernel(X1_arr, X2_arr)
        
    def query_no_data(self, query_points: List[Position]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query GP at points with no data.
        
        Args:
            query_points: Points to query
            
        Returns:
            Tuple of (mean, variance, covariance matrix)
        """
        # Store query points
        self.X_query = query_points
        
        # Compute mean at query points
        self.mXq = np.array([self.m(x) for x in query_points])
        
        # Compute kernel matrix for query points
        self.KXqXq = self._kernel_matrix(query_points, query_points)
        
        # Add small constant for numerical stability
        νₚ = np.diag(self.KXqXq) + np.finfo(float).eps
        
        return self.mXq, νₚ, self.KXqXq
        
    def query(self, query_points: List[Position]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query GP at points.
        
        Args:
            query_points: Points to query
            
        Returns:
            Tuple of (mean, variance, covariance matrix)
        """
        if not self.X:
            return self.query_no_data(query_points)
            
        # Compute kernel matrices
        self.KXqX = self._kernel_matrix(query_points, self.X)
        self.KXqXq = self._kernel_matrix(query_points, query_points)
        
        # Compute mean at query points
        self.mXq = np.array([self.m(x) for x in query_points])
        
        # Add noise to kernel matrix
        KXX_noisy = self.KXX + np.diag(self.ν)
        
        # Solve linear system using Cholesky decomposition for numerical stability
        try:
            L = cholesky(KXX_noisy, lower=True)
            alpha = solve_triangular(L.T, solve_triangular(L, self.y - np.array([self.m(x) for x in self.X]), lower=True))
            
            # Compute posterior mean and covariance
            μₚ = self.mXq + self.KXqX @ alpha
            v = solve_triangular(L, self.KXqX.T, lower=True)
            S = self.KXqXq - v.T @ v
            
            # Add small constant for numerical stability
            νₚ = np.diag(S) + np.finfo(float).eps
            
            return μₚ, νₚ, S
        except np.linalg.LinAlgError:
            # Handle numerical instability by adding jitter
            KXX_noisy = KXX_noisy + np.eye(len(self.X)) * 1e-6
            L = cholesky(KXX_noisy, lower=True)
            alpha = solve_triangular(L.T, solve_triangular(L, self.y - np.array([self.m(x) for x in self.X]), lower=True))
            
            # Compute posterior mean and covariance
            μₚ = self.mXq + self.KXqX @ alpha
            v = solve_triangular(L, self.KXqX.T, lower=True)
            S = self.KXqXq - v.T @ v
            
            # Add small constant for numerical stability
            νₚ = np.diag(S) + np.finfo(float).eps
            
            return μₚ, νₚ, S
        
    def posterior(self, sample_locations: List[Position], sample_values: List[float], noise_variances: List[float]) -> None:
        """Update GP with new samples.
        
        Args:
            sample_locations: Sample locations
            sample_values: Sample values
            noise_variances: Noise variances for samples
        """
        if not self.X:
            # First samples
            self.X = sample_locations
            self.y = sample_values
            self.ν = noise_variances
            self.KXX = self._kernel_matrix(sample_locations, sample_locations)
            if self.X_query:
                self.KXqX = self._kernel_matrix(self.X_query, sample_locations)
        else:
            # Update with new samples
            a = self._kernel_matrix(self.X, sample_locations)
            self.KXX = np.block([[self.KXX, a], [a.T, self._kernel_matrix(sample_locations, sample_locations)]])
            if self.X_query:
                new_KXqX = self._kernel_matrix(self.X_query, sample_locations)
                self.KXqX = np.hstack([self.KXqX, new_KXqX])
            
            # Update data
            self.X.extend(sample_locations)
            self.y.extend(sample_values)
            self.ν.extend(noise_variances) 