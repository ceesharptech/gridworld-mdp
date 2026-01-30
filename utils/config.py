"""
Configuration Module

This module defines default parameters and configuration constants
for the Grid World MDP application.

Author: AI Research Project
Date: January 2026
"""

from typing import Tuple, Set, Dict

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Default grid sizes
DEFAULT_GRID_SIZE: Tuple[int, int] = (5, 5)
SMALL_GRID: Tuple[int, int] = (5, 5)
MEDIUM_GRID: Tuple[int, int] = (10, 10)
LARGE_GRID: Tuple[int, int] = (15, 15)

# Default positions
DEFAULT_START: Tuple[int, int] = (0, 0)
DEFAULT_GOAL: Tuple[int, int] = (4, 4)

# Default obstacles for 5x5 grid
DEFAULT_OBSTACLES_5X5: Set[Tuple[int, int]] = {
    (1, 1), (1, 3), (3, 1), (3, 3)
}

# Default obstacles for 10x10 grid
DEFAULT_OBSTACLES_10X10: Set[Tuple[int, int]] = {
    (2, 2), (2, 3), (2, 4),
    (5, 5), (5, 6), (6, 5), (6, 6),
    (7, 2), (8, 2), (9, 2)
}

# Special rewards (optional)
DEFAULT_SPECIAL_REWARDS: Dict[Tuple[int, int], float] = {}

# ============================================================================
# ALGORITHM CONFIGURATION
# ============================================================================

# Value Iteration parameters
DEFAULT_GAMMA: float = 0.9
DEFAULT_THETA: float = 1e-6
DEFAULT_MAX_ITERATIONS: int = 1000

# Gamma range for experiments
GAMMA_RANGE: list = [0.5, 0.7, 0.9, 0.95, 0.99]

# Convergence thresholds for analysis
THETA_RANGE: list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# ============================================================================
# TRANSITION CONFIGURATION
# ============================================================================

# Transition modes
DETERMINISTIC: bool = False
STOCHASTIC: bool = True

# Stochastic transition parameters
DEFAULT_SUCCESS_PROB: float = 0.8
SUCCESS_PROB_RANGE: list = [0.6, 0.7, 0.8, 0.9]

# ============================================================================
# REWARD CONFIGURATION
# ============================================================================

# Default rewards
DEFAULT_GOAL_REWARD: float = 100.0
DEFAULT_STEP_PENALTY: float = -1.0
DEFAULT_OBSTACLE_PENALTY: float = -10.0

# Reward ranges for experiments
GOAL_REWARD_RANGE: list = [50.0, 100.0, 200.0]
STEP_PENALTY_RANGE: list = [-0.5, -1.0, -2.0]

# ============================================================================
# SIMULATION CONFIGURATION
# ============================================================================

# Episode parameters
DEFAULT_MAX_STEPS: int = 1000
DEFAULT_NUM_EPISODES: int = 10

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Figure sizes
SMALL_FIG_SIZE: Tuple[int, int] = (8, 8)
MEDIUM_FIG_SIZE: Tuple[int, int] = (10, 10)
LARGE_FIG_SIZE: Tuple[int, int] = (12, 12)

# Colormaps
VALUE_COLORMAP: str = 'RdYlGn'
HEATMAP_COLORMAP: str = 'viridis'

# Display options
SHOW_VALUES: bool = True
SHOW_COORDINATES: bool = False

# ============================================================================
# PREDEFINED SCENARIOS
# ============================================================================

SCENARIOS = {
    'simple': {
        'grid_size': (5, 5),
        'start_state': (0, 0),
        'goal_state': (4, 4),
        'obstacles': {(2, 2)},
        'description': 'Simple 5x5 grid with one obstacle'
    },
    'corridor': {
        'grid_size': (7, 7),
        'start_state': (0, 0),
        'goal_state': (6, 6),
        'obstacles': {(3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6)},
        'description': 'Corridor maze requiring path finding'
    },
    'complex': {
        'grid_size': (10, 10),
        'start_state': (0, 0),
        'goal_state': (9, 9),
        'obstacles': DEFAULT_OBSTACLES_10X10,
        'description': 'Complex 10x10 grid with multiple obstacles'
    },
    'open': {
        'grid_size': (8, 8),
        'start_state': (0, 0),
        'goal_state': (7, 7),
        'obstacles': set(),
        'description': 'Open grid with no obstacles'
    }
}

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENT_CONFIGS = {
    'gamma_sweep': {
        'gamma_values': GAMMA_RANGE,
        'num_trials': 5
    },
    'stochastic_comparison': {
        'success_probs': SUCCESS_PROB_RANGE,
        'num_episodes': 10
    },
    'grid_size_scaling': {
        'grid_sizes': [(5, 5), (7, 7), (10, 10), (12, 12), (15, 15)]
    },
    'reward_structure': {
        'configs': [
            {'name': 'high_goal', 'goal_reward': 200.0, 'step_penalty': -1.0},
            {'name': 'low_penalty', 'goal_reward': 100.0, 'step_penalty': -0.5},
            {'name': 'high_penalty', 'goal_reward': 100.0, 'step_penalty': -2.0},
        ]
    }
}

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

# Streamlit configuration
APP_TITLE: str = "Grid World MDP - Value Iteration"
APP_ICON: str = "🤖"
PAGE_LAYOUT: str = "wide"

# Sidebar defaults
SIDEBAR_WIDTH: int = 400

# Performance settings
ENABLE_CACHING: bool = True
CACHE_TTL: int = 3600  # seconds

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_scenario(name: str) -> Dict:
    """
    Get predefined scenario configuration.
    
    Args:
        name: Scenario name
        
    Returns:
        Scenario configuration dictionary
    """
    return SCENARIOS.get(name, SCENARIOS['simple'])


def get_default_config() -> Dict:
    """
    Get complete default configuration.
    
    Returns:
        Dictionary with all default parameters
    """
    return {
        'environment': {
            'grid_size': DEFAULT_GRID_SIZE,
            'start_state': DEFAULT_START,
            'goal_state': DEFAULT_GOAL,
            'obstacles': DEFAULT_OBSTACLES_5X5,
            'special_rewards': DEFAULT_SPECIAL_REWARDS
        },
        'algorithm': {
            'gamma': DEFAULT_GAMMA,
            'theta': DEFAULT_THETA,
            'max_iterations': DEFAULT_MAX_ITERATIONS
        },
        'transition': {
            'stochastic': False,
            'success_prob': DEFAULT_SUCCESS_PROB
        },
        'reward': {
            'goal_reward': DEFAULT_GOAL_REWARD,
            'step_penalty': DEFAULT_STEP_PENALTY,
            'obstacle_penalty': DEFAULT_OBSTACLE_PENALTY
        },
        'simulation': {
            'max_steps': DEFAULT_MAX_STEPS,
            'num_episodes': DEFAULT_NUM_EPISODES
        }
    }


def validate_gamma(gamma: float) -> bool:
    """Validate discount factor."""
    return 0 < gamma <= 1.0


def validate_probability(prob: float) -> bool:
    """Validate probability value."""
    return 0 <= prob <= 1.0


def validate_grid_size(size: Tuple[int, int]) -> bool:
    """Validate grid size."""
    rows, cols = size
    return rows > 0 and cols > 0 and rows <= 50 and cols <= 50
