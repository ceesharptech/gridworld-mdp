"""
Value Iteration Algorithm Module

This module implements the Value Iteration algorithm for solving MDPs.
Value Iteration computes the optimal value function V*(s) through iterative
application of the Bellman optimality equation.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
from typing import Dict, Tuple, List
from mdp.environment import GridWorld
from mdp.transition import TransitionModel
from mdp.rewards import RewardFunction


class ValueIteration:
    """
    Value Iteration algorithm for MDP optimal policy computation.
    
    Implements the Bellman optimality equation iteratively:
    V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]
    
    The algorithm converges when the maximum change in value function
    falls below a threshold ε.
    
    Attributes:
        env (GridWorld): The environment
        transition_model (TransitionModel): Transition probability model
        reward_function (RewardFunction): Reward function
        gamma (float): Discount factor
        theta (float): Convergence threshold
        max_iterations (int): Maximum iterations before stopping
    """
    
    def __init__(
        self,
        env: GridWorld,
        transition_model: TransitionModel,
        reward_function: RewardFunction,
        gamma: float = 0.9,
        theta: float = 1e-6,
        max_iterations: int = 1000
    ):
        """
        Initialize Value Iteration solver.
        
        Args:
            env: Grid World environment
            transition_model: Transition probability model
            reward_function: Reward function
            gamma: Discount factor (0 < γ ≤ 1)
            theta: Convergence threshold (ε)
            max_iterations: Maximum number of iterations
        """
        self.env = env
        self.transition_model = transition_model
        self.reward_function = reward_function
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Initialize value function
        self.V = {state: 0.0 for state in env.states}
        
        # Iteration tracking
        self.iteration_count = 0
        self.convergence_history = []
        self.converged = False
    
    def compute_q_value(
        self,
        state: Tuple[int, int],
        action: int
    ) -> float:
        """
        Compute Q-value for a state-action pair.
        
        Q(s,a) = Σ P(s'|s,a)[R(s,a,s') + γV(s')]
        
        Args:
            state: Current state (row, col)
            action: Action to evaluate
            
        Returns:
            Q-value for the state-action pair
        """
        q_value = 0.0
        
        # Get all possible next states and their probabilities
        transitions = self.transition_model.get_possible_transitions(state, action)
        
        for next_state, prob in transitions:
            # Get immediate reward
            reward = self.reward_function.get_reward(state, action, next_state)
            
            # Add discounted future value
            q_value += prob * (reward + self.gamma * self.V[next_state])
        
        return q_value
    
    def compute_state_value(self, state: Tuple[int, int]) -> float:
        """
        Compute optimal value for a state using Bellman optimality.
        
        V(s) = max_a Q(s,a)
        
        Args:
            state: State position (row, col)
            
        Returns:
            Optimal value for the state
        """
        # Terminal states have value 0
        if self.env.is_terminal(state):
            return 0.0
        
        # Compute Q-value for each action and take maximum
        q_values = [self.compute_q_value(state, action) 
                   for action in self.env.actions]
        
        return max(q_values)
    
    def iterate(self) -> float:
        """
        Perform one iteration of value iteration.
        
        Updates all state values according to Bellman optimality equation.
        
        Returns:
            Maximum change in value function (delta)
        """
        delta = 0.0
        V_new = {}
        
        # Update value for each state
        for state in self.env.states:
            old_value = self.V[state]
            new_value = self.compute_state_value(state)
            V_new[state] = new_value
            
            # Track maximum change
            change = abs(old_value - new_value)
            delta = max(delta, change)
        
        # Update value function
        self.V = V_new
        
        return delta
    
    def solve(self, verbose: bool = False) -> Dict:
        """
        Run value iteration until convergence.
        
        Iteratively applies Bellman updates until the maximum change
        falls below threshold theta or max_iterations is reached.
        
        Args:
            verbose: If True, print iteration progress
            
        Returns:
            Dictionary containing:
                - value_function: Final value function V(s)
                - iterations: Number of iterations performed
                - converged: Whether algorithm converged
                - convergence_history: List of delta values per iteration
        """
        self.iteration_count = 0
        self.convergence_history = []
        self.converged = False
        
        if verbose:
            print(f"Starting Value Iteration (γ={self.gamma}, θ={self.theta})")
            print("-" * 60)
        
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            
            # Perform one iteration
            delta = self.iterate()
            self.convergence_history.append(delta)
            
            if verbose and self.iteration_count % 10 == 0:
                print(f"Iteration {self.iteration_count}: δ = {delta:.6f}")
            
            # Check convergence
            if delta < self.theta:
                self.converged = True
                if verbose:
                    print(f"\nConverged after {self.iteration_count} iterations!")
                    print(f"Final δ = {delta:.6f}")
                break
        
        if not self.converged and verbose:
            print(f"\nReached maximum iterations ({self.max_iterations})")
            print(f"Final δ = {delta:.6f} (threshold: {self.theta})")
        
        return {
            'value_function': self.V.copy(),
            'iterations': self.iteration_count,
            'converged': self.converged,
            'convergence_history': self.convergence_history.copy()
        }
    
    def get_value_array(self) -> np.ndarray:
        """
        Convert value function to 2D array for visualization.
        
        Returns:
            2D numpy array of values (rows × cols)
        """
        value_grid = np.zeros(self.env.grid_size)
        
        for state in self.env.states:
            row, col = state
            value_grid[row, col] = self.V[state]
        
        # Set obstacle values to NaN for visualization
        for obs in self.env.obstacles:
            row, col = obs
            value_grid[row, col] = np.nan
        
        return value_grid
    
    def get_optimal_action(self, state: Tuple[int, int]) -> int:
        """
        Get the optimal action for a state based on current value function.
        
        π*(s) = argmax_a Q(s,a)
        
        Args:
            state: State position (row, col)
            
        Returns:
            Optimal action
        """
        if self.env.is_terminal(state):
            return None
        
        # Compute Q-value for each action
        q_values = [(action, self.compute_q_value(state, action)) 
                   for action in self.env.actions]
        
        # Return action with maximum Q-value
        optimal_action = max(q_values, key=lambda x: x[1])[0]
        return optimal_action
    
    def reset(self):
        """Reset value function and iteration counters."""
        self.V = {state: 0.0 for state in self.env.states}
        self.iteration_count = 0
        self.convergence_history = []
        self.converged = False
    
    def __repr__(self) -> str:
        """String representation."""
        status = "Converged" if self.converged else "Not converged"
        return (f"ValueIteration(γ={self.gamma}, iterations={self.iteration_count}, "
                f"status={status})")
