from typing import Tuple

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate the Manhattan distance between two positions.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        int: Manhattan distance between the positions
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) 