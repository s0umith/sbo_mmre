"""Map building utilities for the Rover environment."""

from typing import Tuple, List
import numpy as np

def get_neighbors(pos: Tuple[int, int], map_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get valid neighboring positions.
    
    Args:
        pos: Current position (x, y)
        map_size: Size of the map (width, height)
        
    Returns:
        List of valid neighboring positions
    """
    x, y = pos
    neighbors = [
        (x, y+1),  # up
        (x, y-1),  # down
        (x+1, y),  # right
        (x-1, y)   # left
    ]
    
    valid_neighbors = []
    for nx, ny in neighbors:
        if 0 <= nx < map_size[0] and 0 <= ny < map_size[1]:
            valid_neighbors.append((nx, ny))
            
    return valid_neighbors

def build_map(
    rng: np.random.RandomState,
    num_sample_types: int,
    map_size: Tuple[int, int],
    p_neighbors: float = 0.95
) -> np.ndarray:
    """Build a random map with spatial correlation.
    
    Args:
        rng: Random number generator
        num_sample_types: Number of different sample types
        map_size: Size of the map (width, height)
        p_neighbors: Probability of using neighbor values
        
    Returns:
        Generated map
    """
    # Create sample types from 0 to 1
    sample_types = np.linspace(0, 1, num_sample_types, endpoint=False)
    
    # Initialize random map
    init_map = rng.choice(sample_types, size=map_size)
    new_map = np.zeros(map_size)
    
    # Process each position
    for y in range(map_size[1]):
        for x in range(map_size[0]):
            if x == 0 and y == 0:
                new_map[y, x] = init_map[y, x]
                continue
                
            if rng.random() < p_neighbors:
                # Get valid neighbors that have been processed
                neighbors = get_neighbors((x, y), map_size)
                valid_neighbors = [
                    n for n in neighbors 
                    if n[1] < y or (n[1] == y and n[0] < x)
                ]
                
                if valid_neighbors:
                    # Use mean of processed neighbors
                    neighbor_values = [init_map[ny, nx] for nx, ny in valid_neighbors]
                    new_map[y, x] = np.round(np.mean(neighbor_values), decimals=1)
                else:
                    new_map[y, x] = init_map[y, x]
            else:
                new_map[y, x] = init_map[y, x]
                
    return new_map 