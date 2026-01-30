"""
Grid World Environment Module

This module defines the Grid World environment as a Markov Decision Process (MDP).
It specifies the state space, action space, and environment boundaries.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
from typing import Tuple, List, Set, Optional


class GridWorld:
    """
    Grid World Environment for MDP Navigation.
    
    Represents a discrete grid where an agent navigates from a start state
    to a goal state while avoiding obstacles and accumulating rewards.
    
    Attributes:
        grid_size (tuple): (rows, cols) dimensions of the grid
        start_state (tuple): (row, col) starting position
        goal_state (tuple): (row, col) goal position
        obstacles (set): Set of (row, col) positions that are blocked
        special_rewards (dict): Mapping of (row, col) to reward values
        actions (list): Available actions [UP, DOWN, LEFT, RIGHT]
    """
    
    # Action definitions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    ACTION_NAMES = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
    ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1)
    }
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (5, 5),
        start_state: Tuple[int, int] = (0, 0),
        goal_state: Tuple[int, int] = (4, 4),
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        special_rewards: Optional[dict] = None
    ):
        """
        Initialize the Grid World environment.
        
        Args:
            grid_size: Tuple of (rows, cols)
            start_state: Starting position (row, col)
            goal_state: Goal position (row, col)
            obstacles: Set of obstacle positions
            special_rewards: Dictionary mapping positions to special rewards
        
        Raises:
            ValueError: If start or goal states are invalid
        """
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        
        # Validate states
        if not self._is_valid_state(start_state):
            raise ValueError(f"Invalid start state: {start_state}")
        if not self._is_valid_state(goal_state):
            raise ValueError(f"Invalid goal state: {goal_state}")
        
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacles = obstacles if obstacles is not None else set()
        self.special_rewards = special_rewards if special_rewards is not None else {}
        
        # Ensure start and goal are not obstacles
        if self.start_state in self.obstacles:
            self.obstacles.remove(self.start_state)
        if self.goal_state in self.obstacles:
            self.obstacles.remove(self.goal_state)
        
        # Define action space
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        self.n_actions = len(self.actions)
        
        # Generate state space
        self.states = self._generate_state_space()
        self.n_states = len(self.states)
    
    def _is_valid_state(self, state: Tuple[int, int]) -> bool:
        """
        Check if a state is within grid boundaries.
        
        Args:
            state: (row, col) position
            
        Returns:
            True if state is valid, False otherwise
        """
        row, col = state
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _generate_state_space(self) -> List[Tuple[int, int]]:
        """
        Generate all valid states in the grid.
        
        Returns:
            List of all valid (row, col) states
        """
        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                if state not in self.obstacles:
                    states.append(state)
        return states
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """
        Check if a state is terminal (goal state).
        
        Args:
            state: (row, col) position
            
        Returns:
            True if state is the goal, False otherwise
        """
        return state == self.goal_state
    
    def is_obstacle(self, state: Tuple[int, int]) -> bool:
        """
        Check if a state is an obstacle.
        
        Args:
            state: (row, col) position
            
        Returns:
            True if state is an obstacle, False otherwise
        """
        return state in self.obstacles
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Get the resulting state from taking an action (deterministic).
        
        This method returns the next state assuming perfect execution of the action.
        If the resulting state is invalid (obstacle or out of bounds), returns current state.
        
        Args:
            state: Current (row, col) position
            action: Action to take (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            Next (row, col) position
        """
        if action not in self.ACTION_DELTAS:
            return state
        
        delta_row, delta_col = self.ACTION_DELTAS[action]
        next_row = state[0] + delta_row
        next_col = state[1] + delta_col
        next_state = (next_row, next_col)
        
        # Check if next state is valid
        if not self._is_valid_state(next_state) or self.is_obstacle(next_state):
            return state  # Stay in place if invalid move
        
        return next_state
    
    def get_action_name(self, action: int) -> str:
        """
        Get the symbol representation of an action.
        
        Args:
            action: Action integer
            
        Returns:
            Arrow symbol representing the action
        """
        return self.ACTION_NAMES.get(action, "?")
    
    def get_state_info(self, state: Tuple[int, int]) -> dict:
        """
        Get information about a specific state.
        
        Args:
            state: (row, col) position
            
        Returns:
            Dictionary with state information
        """
        return {
            "state": state,
            "is_start": state == self.start_state,
            "is_goal": state == self.goal_state,
            "is_terminal": self.is_terminal(state),
            "is_obstacle": self.is_obstacle(state),
            "special_reward": self.special_rewards.get(state, None)
        }
    
    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to the start state.
        
        Returns:
            Start state position
        """
        return self.start_state
    
    def __repr__(self) -> str:
        """String representation of the environment."""
        return (f"GridWorld({self.rows}x{self.cols}, "
                f"start={self.start_state}, goal={self.goal_state}, "
                f"obstacles={len(self.obstacles)})")
