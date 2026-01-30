"""
Policy Extraction Module

This module extracts the optimal policy from a computed value function.
After Value Iteration converges, the optimal policy is derived by selecting
the action that maximizes the expected value at each state.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
from typing import Dict, Tuple, Optional
from mdp.environment import GridWorld
from mdp.transition import TransitionModel
from mdp.rewards import RewardFunction


class PolicyExtractor:
    """
    Extracts optimal policy from value function.
    
    Given a value function V(s), computes the optimal policy:
    π*(s) = argmax_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]
    
    The policy maps each state to the best action to take.
    
    Attributes:
        env (GridWorld): The environment
        transition_model (TransitionModel): Transition probability model
        reward_function (RewardFunction): Reward function
        gamma (float): Discount factor
    """
    
    def __init__(
        self,
        env: GridWorld,
        transition_model: TransitionModel,
        reward_function: RewardFunction,
        gamma: float = 0.9
    ):
        """
        Initialize policy extractor.
        
        Args:
            env: Grid World environment
            transition_model: Transition probability model
            reward_function: Reward function
            gamma: Discount factor
        """
        self.env = env
        self.transition_model = transition_model
        self.reward_function = reward_function
        self.gamma = gamma
        
        self.policy = {}
        self.q_values = {}
    
    def compute_q_value(
        self,
        state: Tuple[int, int],
        action: int,
        value_function: Dict[Tuple[int, int], float]
    ) -> float:
        """
        Compute Q-value for a state-action pair given value function.
        
        Q(s,a) = Σ P(s'|s,a)[R(s,a,s') + γV(s')]
        
        Args:
            state: Current state (row, col)
            action: Action to evaluate
            value_function: Dictionary mapping states to values
            
        Returns:
            Q-value for the state-action pair
        """
        q_value = 0.0
        
        transitions = self.transition_model.get_possible_transitions(state, action)
        
        for next_state, prob in transitions:
            reward = self.reward_function.get_reward(state, action, next_state)
            future_value = value_function.get(next_state, 0.0)
            q_value += prob * (reward + self.gamma * future_value)
        
        return q_value
    
    def extract_policy(
        self,
        value_function: Dict[Tuple[int, int], float]
    ) -> Dict[Tuple[int, int], int]:
        """
        Extract optimal policy from value function.
        
        For each state, selects the action that maximizes expected value.
        
        Args:
            value_function: Dictionary mapping states to values
            
        Returns:
            Dictionary mapping states to optimal actions
        """
        self.policy = {}
        self.q_values = {}
        
        for state in self.env.states:
            # Terminal states have no action
            if self.env.is_terminal(state):
                self.policy[state] = None
                continue
            
            # Compute Q-value for each action
            action_values = {}
            for action in self.env.actions:
                q_val = self.compute_q_value(state, action, value_function)
                action_values[action] = q_val
            
            # Store Q-values for this state
            self.q_values[state] = action_values
            
            # Select action with maximum Q-value
            best_action = max(action_values.items(), key=lambda x: x[1])[0]
            self.policy[state] = best_action
        
        return self.policy
    
    def get_policy_array(self) -> np.ndarray:
        """
        Convert policy to 2D array of action indices for visualization.
        
        Returns:
            2D numpy array of action indices (rows × cols)
            -1 for obstacles and terminal states
        """
        policy_grid = np.full(self.env.grid_size, -1, dtype=int)
        
        for state, action in self.policy.items():
            row, col = state
            if action is not None:
                policy_grid[row, col] = action
        
        return policy_grid
    
    def get_policy_arrows(self) -> Dict[Tuple[int, int], str]:
        """
        Get policy as arrow symbols for display.
        
        Returns:
            Dictionary mapping states to arrow strings (↑, ↓, ←, →)
        """
        arrows = {}
        for state, action in self.policy.items():
            if action is not None:
                arrows[state] = self.env.get_action_name(action)
            else:
                arrows[state] = "G" if self.env.is_terminal(state) else " "
        
        return arrows
    
    def get_action_for_state(self, state: Tuple[int, int]) -> Optional[int]:
        """
        Get the optimal action for a specific state.
        
        Args:
            state: State position (row, col)
            
        Returns:
            Optimal action, or None if terminal
        """
        return self.policy.get(state, None)
    
    def get_q_values_for_state(
        self,
        state: Tuple[int, int]
    ) -> Optional[Dict[int, float]]:
        """
        Get Q-values for all actions at a specific state.
        
        Args:
            state: State position (row, col)
            
        Returns:
            Dictionary mapping actions to Q-values, or None if terminal
        """
        return self.q_values.get(state, None)
    
    def policy_evaluation(
        self,
        value_function: Dict[Tuple[int, int], float],
        num_states: int = None
    ) -> float:
        """
        Compute average value under the extracted policy.
        
        This provides a measure of policy quality.
        
        Args:
            value_function: Value function used for policy extraction
            num_states: Number of states to average over (default: all non-terminal)
            
        Returns:
            Average state value under policy
        """
        non_terminal_states = [s for s in self.env.states 
                              if not self.env.is_terminal(s)]
        
        if num_states is None:
            states_to_evaluate = non_terminal_states
        else:
            states_to_evaluate = non_terminal_states[:num_states]
        
        if not states_to_evaluate:
            return 0.0
        
        total_value = sum(value_function.get(s, 0.0) for s in states_to_evaluate)
        return total_value / len(states_to_evaluate)
    
    def is_deterministic(self) -> bool:
        """
        Check if the policy is deterministic (one action per state).
        
        Returns:
            True (policies extracted this way are always deterministic)
        """
        return True
    
    def get_policy_stats(self) -> Dict:
        """
        Get statistics about the policy.
        
        Returns:
            Dictionary with policy statistics
        """
        action_counts = {action: 0 for action in self.env.actions}
        terminal_count = 0
        
        for state, action in self.policy.items():
            if action is None:
                terminal_count += 1
            else:
                action_counts[action] += 1
        
        return {
            'total_states': len(self.policy),
            'terminal_states': terminal_count,
            'action_distribution': action_counts,
            'non_terminal_states': len(self.policy) - terminal_count
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PolicyExtractor(states={len(self.policy)})"
