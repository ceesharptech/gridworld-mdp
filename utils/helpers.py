"""
Helpers Module

This module provides utility functions for common operations
throughout the Grid World MDP application.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
from typing import Tuple, List, Dict, Set, Any
import json
import pickle


# ============================================================================
# COORDINATE UTILITIES
# ============================================================================

def manhattan_distance(state1: Tuple[int, int], state2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two states.
    
    Args:
        state1: First state (row, col)
        state2: Second state (row, col)
        
    Returns:
        Manhattan distance
    """
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])


def euclidean_distance(state1: Tuple[int, int], state2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two states.
    
    Args:
        state1: First state (row, col)
        state2: Second state (row, col)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)


def get_neighbors(
    state: Tuple[int, int],
    grid_size: Tuple[int, int],
    include_diagonal: bool = False
) -> List[Tuple[int, int]]:
    """
    Get valid neighboring states.
    
    Args:
        state: Current state (row, col)
        grid_size: Grid dimensions (rows, cols)
        include_diagonal: Whether to include diagonal neighbors
        
    Returns:
        List of valid neighbor states
    """
    row, col = state
    rows, cols = grid_size
    
    neighbors = []
    
    # Cardinal directions
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Add diagonal directions if requested
    if include_diagonal:
        deltas.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    
    for dr, dc in deltas:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbors.append((new_row, new_col))
    
    return neighbors


# ============================================================================
# DATA STRUCTURE UTILITIES
# ============================================================================

def state_dict_to_array(
    state_dict: Dict[Tuple[int, int], Any],
    grid_size: Tuple[int, int],
    default_value: Any = 0
) -> np.ndarray:
    """
    Convert state dictionary to 2D numpy array.
    
    Args:
        state_dict: Dictionary mapping states to values
        grid_size: Grid dimensions
        default_value: Value for missing states
        
    Returns:
        2D numpy array
    """
    rows, cols = grid_size
    array = np.full((rows, cols), default_value)
    
    for (row, col), value in state_dict.items():
        array[row, col] = value
    
    return array


def array_to_state_dict(
    array: np.ndarray,
    exclude_value: Any = None
) -> Dict[Tuple[int, int], Any]:
    """
    Convert 2D numpy array to state dictionary.
    
    Args:
        array: 2D numpy array
        exclude_value: Value to exclude from dictionary
        
    Returns:
        Dictionary mapping states to values
    """
    state_dict = {}
    rows, cols = array.shape
    
    for row in range(rows):
        for col in range(cols):
            value = array[row, col]
            if exclude_value is None or value != exclude_value:
                state_dict[(row, col)] = value
    
    return state_dict


# ============================================================================
# PATH UTILITIES
# ============================================================================

def calculate_path_length(path: List[Tuple[int, int]]) -> int:
    """
    Calculate the length of a path (number of steps).
    
    Args:
        path: List of states
        
    Returns:
        Number of steps (transitions)
    """
    return len(path) - 1 if len(path) > 0 else 0


def has_cycles(path: List[Tuple[int, int]]) -> bool:
    """
    Check if a path contains cycles (repeated states).
    
    Args:
        path: List of states
        
    Returns:
        True if path has cycles
    """
    return len(path) != len(set(path))


def remove_cycles(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Remove cycles from a path by keeping only the last occurrence.
    
    Args:
        path: List of states
        
    Returns:
        Path without cycles
    """
    if not has_cycles(path):
        return path
    
    # Build path without cycles
    seen = {}
    for i, state in enumerate(path):
        seen[state] = i
    
    # Reconstruct path
    result = []
    current_idx = 0
    while current_idx < len(path):
        state = path[current_idx]
        result.append(state)
        
        # Jump to last occurrence if there's a cycle
        last_idx = seen[state]
        if last_idx > current_idx:
            current_idx = last_idx + 1
        else:
            current_idx += 1
    
    return result


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_state(state: Tuple[int, int]) -> str:
    """
    Format state coordinates for display.
    
    Args:
        state: State tuple (row, col)
        
    Returns:
        Formatted string
    """
    return f"({state[0]}, {state[1]})"


def format_time(seconds: float) -> str:
    """
    Format time duration.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def is_valid_state(state: Tuple[int, int], grid_size: Tuple[int, int]) -> bool:
    """
    Check if a state is within grid bounds.
    
    Args:
        state: State position
        grid_size: Grid dimensions
        
    Returns:
        True if valid
    """
    row, col = state
    rows, cols = grid_size
    return 0 <= row < rows and 0 <= col < cols


def is_valid_path(
    path: List[Tuple[int, int]],
    grid_size: Tuple[int, int],
    obstacles: Set[Tuple[int, int]]
) -> bool:
    """
    Check if a path is valid (all states in bounds and not obstacles).
    
    Args:
        path: List of states
        grid_size: Grid dimensions
        obstacles: Set of obstacle positions
        
    Returns:
        True if path is valid
    """
    for state in path:
        if not is_valid_state(state, grid_size):
            return False
        if state in obstacles:
            return False
    
    return True


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }


def normalize_values(values: List[float]) -> List[float]:
    """
    Normalize values to [0, 1] range.
    
    Args:
        values: List of values
        
    Returns:
        Normalized values
    """
    if not values:
        return []
    
    arr = np.array(values)
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if max_val == min_val:
        return [0.5] * len(values)
    
    return ((arr - min_val) / (max_val - min_val)).tolist()


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_to_json(data: Any, filename: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filename: Output filename
    """
    # Convert tuples to lists for JSON serialization
    serializable_data = convert_tuples_to_lists(data)
    
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_from_json(filename: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Loaded data
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to tuples where appropriate
    return convert_lists_to_tuples(data)


def convert_tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists for JSON serialization."""
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, tuple) else k: convert_tuples_to_lists(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_tuples_to_lists(item) for item in obj]
    else:
        return obj


def convert_lists_to_tuples(obj: Any, tuple_keys: bool = True) -> Any:
    """Recursively convert lists to tuples where appropriate."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Try to convert string keys that look like tuples
            if tuple_keys and isinstance(k, str) and k.startswith('('):
                try:
                    key = eval(k)
                    if isinstance(key, tuple):
                        result[key] = convert_lists_to_tuples(v)
                    else:
                        result[k] = convert_lists_to_tuples(v)
                except:
                    result[k] = convert_lists_to_tuples(v)
            else:
                result[k] = convert_lists_to_tuples(v)
        return result
    elif isinstance(obj, list):
        return [convert_lists_to_tuples(item, False) for item in obj]
    else:
        return obj


# ============================================================================
# RANDOM UTILITIES
# ============================================================================

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    import random
    random.seed(seed)


def generate_random_obstacles(
    grid_size: Tuple[int, int],
    num_obstacles: int,
    start_state: Tuple[int, int],
    goal_state: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """
    Generate random obstacle positions.
    
    Args:
        grid_size: Grid dimensions
        num_obstacles: Number of obstacles to generate
        start_state: Start position (excluded)
        goal_state: Goal position (excluded)
        
    Returns:
        Set of obstacle positions
    """
    rows, cols = grid_size
    all_states = [(r, c) for r in range(rows) for c in range(cols)]
    
    # Remove start and goal
    available = [s for s in all_states if s != start_state and s != goal_state]
    
    # Sample random obstacles
    num_obstacles = min(num_obstacles, len(available))
    obstacles = set(np.random.choice(
        len(available), num_obstacles, replace=False
    ))
    
    return {available[i] for i in obstacles}
