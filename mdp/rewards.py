"""
Reward Function Module

This module implements the reward function R(s,a,s') for the MDP.
Provides configurable rewards for different state transitions.

Author: AI Research Project
Date: January 2026
"""

from typing import Tuple, Dict
from mdp.environment import GridWorld


class RewardFunction:
    """
    Reward function for Grid World MDP.
    
    Implements R(s,a,s'): immediate reward received when transitioning
    from state s to state s' via action a.
    
    Reward structure:
    - Goal reward: Large positive reward for reaching goal
    - Step penalty: Small negative reward for each step (encourages shorter paths)
    - Special rewards: Custom rewards for specific cells
    - Obstacle penalty: Optional penalty for attempting to enter obstacles
    
    Attributes:
        env (GridWorld): The environment
        goal_reward (float): Reward for reaching goal state
        step_penalty (float): Penalty for each step taken
        obstacle_penalty (float): Penalty for hitting obstacles
    """
    
    def __init__(
        self,
        env: GridWorld,
        goal_reward: float = 100.0,
        step_penalty: float = -1.0,
        obstacle_penalty: float = -10.0
    ):
        """
        Initialize the reward function.
        
        Args:
            env: Grid World environment
            goal_reward: Reward for reaching the goal state
            step_penalty: Penalty applied to each non-goal transition
            obstacle_penalty: Penalty for attempting to move into obstacle
        """
        self.env = env
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
    
    def get_reward(
        self,
        state: Tuple[int, int],
        action: int,
        next_state: Tuple[int, int]
    ) -> float:
        """
        Get the immediate reward for a state transition.
        
        R(s,a,s')
        
        Args:
            state: Current state (row, col)
            action: Action taken
            next_state: Resulting state (row, col)
            
        Returns:
            Immediate reward (float)
        """
        # Goal reached
        if next_state == self.env.goal_state:
            return self.goal_reward
        
        # Check for special rewards at next_state
        if next_state in self.env.special_rewards:
            return self.env.special_rewards[next_state]
        
        # Attempted to move into obstacle (stayed in place)
        if state == next_state and not self.env.is_terminal(state):
            # Check if action would have led to obstacle or boundary
            intended_next = self.env.get_next_state(state, action)
            if intended_next == state:  # Movement was blocked
                return self.obstacle_penalty
        
        # Default: step penalty
        return self.step_penalty
    
    def get_expected_reward(
        self,
        state: Tuple[int, int],
        action: int,
        transition_model
    ) -> float:
        """
        Calculate expected immediate reward for a state-action pair.
        
        E[R(s,a)] = Σ P(s'|s,a) * R(s,a,s')
        
        This is useful for stochastic environments where multiple
        next states are possible.
        
        Args:
            state: Current state
            action: Action taken
            transition_model: TransitionModel instance
            
        Returns:
            Expected reward (float)
        """
        expected_reward = 0.0
        transitions = transition_model.get_possible_transitions(state, action)
        
        for next_state, prob in transitions:
            reward = self.get_reward(state, action, next_state)
            expected_reward += prob * reward
        
        return expected_reward
    
    def get_state_reward_info(self, state: Tuple[int, int]) -> Dict:
        """
        Get reward information for a specific state.
        
        Args:
            state: State position (row, col)
            
        Returns:
            Dictionary with reward information
        """
        info = {
            "state": state,
            "is_goal": state == self.env.goal_state,
        }
        
        if state == self.env.goal_state:
            info["reward_type"] = "goal"
            info["reward_value"] = self.goal_reward
        elif state in self.env.special_rewards:
            info["reward_type"] = "special"
            info["reward_value"] = self.env.special_rewards[state]
        else:
            info["reward_type"] = "step"
            info["reward_value"] = self.step_penalty
        
        return info
    
    def update_parameters(
        self,
        goal_reward: float = None,
        step_penalty: float = None,
        obstacle_penalty: float = None
    ):
        """
        Update reward function parameters.
        
        Args:
            goal_reward: New goal reward (optional)
            step_penalty: New step penalty (optional)
            obstacle_penalty: New obstacle penalty (optional)
        """
        if goal_reward is not None:
            self.goal_reward = goal_reward
        if step_penalty is not None:
            self.step_penalty = step_penalty
        if obstacle_penalty is not None:
            self.obstacle_penalty = obstacle_penalty
    
    def get_total_possible_reward(self, min_steps: int) -> float:
        """
        Calculate theoretical maximum reward.
        
        This assumes optimal path with no special penalties.
        
        Args:
            min_steps: Minimum steps to reach goal
            
        Returns:
            Maximum possible total reward
        """
        # Goal reward minus step penalties for optimal path
        return self.goal_reward + (min_steps * self.step_penalty)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"RewardFunction(goal={self.goal_reward}, "
                f"step={self.step_penalty}, obstacle={self.obstacle_penalty})")
