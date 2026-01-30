"""
Agent Module

This module implements an agent that can execute policies in the Grid World.
The agent follows a policy to navigate from start to goal.

Author: AI Research Project
Date: January 2026
"""

import random
from typing import Tuple, Optional, Dict, List
from mdp.environment import GridWorld


class Agent:
    """
    Agent that executes policies in Grid World.
    
    The agent can follow:
    - Optimal policies (from Value Iteration)
    - Random policies (baseline)
    - Custom policies
    
    Attributes:
        env (GridWorld): The environment
        policy (dict): Mapping from states to actions
        current_state (tuple): Current position (row, col)
        start_state (tuple): Starting position
    """
    
    def __init__(
        self,
        env: GridWorld,
        policy: Optional[Dict[Tuple[int, int], int]] = None
    ):
        """
        Initialize the agent.
        
        Args:
            env: Grid World environment
            policy: Optional policy dictionary (state -> action)
                   If None, agent uses random policy
        """
        self.env = env
        self.policy = policy
        self.current_state = env.start_state
        self.start_state = env.start_state
        
        # Track history
        self.state_history = [self.current_state]
        self.action_history = []
        self.reward_history = []
    
    def select_action(self, state: Tuple[int, int]) -> int:
        """
        Select an action for the given state.
        
        If policy is provided, uses policy.
        Otherwise, selects random action.
        
        Args:
            state: Current state (row, col)
            
        Returns:
            Action to take
        """
        # Check if terminal
        if self.env.is_terminal(state):
            return None
        
        # Use policy if available
        if self.policy is not None and state in self.policy:
            action = self.policy[state]
            # Policy might have None for terminal states
            if action is not None:
                return action
        
        # Random action as fallback
        return random.choice(self.env.actions)
    
    def execute_action(
        self,
        action: int,
        transition_model=None
    ) -> Tuple[Tuple[int, int], float]:
        """
        Execute an action and transition to next state.
        
        Args:
            action: Action to execute
            transition_model: Optional transition model for stochastic moves
                            If None, uses deterministic transitions
            
        Returns:
            Tuple of (next_state, reward)
        """
        current = self.current_state
        
        # Determine next state
        if transition_model is not None:
            # Stochastic transition
            transitions = transition_model.get_possible_transitions(current, action)
            # Sample based on probabilities
            states, probs = zip(*transitions)
            next_state = random.choices(states, weights=probs, k=1)[0]
        else:
            # Deterministic transition
            next_state = self.env.get_next_state(current, action)
        
        # For reward calculation, we need the reward function
        # Since Agent doesn't have direct access, we'll return 0 here
        # The simulator will calculate proper rewards
        reward = 0.0
        
        # Update state
        self.current_state = next_state
        
        return next_state, reward
    
    def step(
        self,
        transition_model=None
    ) -> Tuple[Tuple[int, int], int, bool]:
        """
        Execute one step: select action and move.
        
        Args:
            transition_model: Optional transition model
            
        Returns:
            Tuple of (next_state, action_taken, is_terminal)
        """
        # Select action
        action = self.select_action(self.current_state)
        
        if action is None:
            # Already at terminal state
            return self.current_state, None, True
        
        # Execute action
        next_state, _ = self.execute_action(action, transition_model)
        
        # Record history
        self.action_history.append(action)
        self.state_history.append(next_state)
        
        # Check if reached terminal
        is_terminal = self.env.is_terminal(next_state)
        
        return next_state, action, is_terminal
    
    def reset(self, start_state: Optional[Tuple[int, int]] = None):
        """
        Reset agent to starting position.
        
        Args:
            start_state: Optional custom start state
        """
        if start_state is not None:
            self.start_state = start_state
        
        self.current_state = self.start_state
        self.state_history = [self.current_state]
        self.action_history = []
        self.reward_history = []
    
    def get_state(self) -> Tuple[int, int]:
        """Get current state."""
        return self.current_state
    
    def get_history(self) -> Dict:
        """
        Get agent's history.
        
        Returns:
            Dictionary with state, action, and reward history
        """
        return {
            'states': self.state_history.copy(),
            'actions': self.action_history.copy(),
            'rewards': self.reward_history.copy()
        }
    
    def set_policy(self, policy: Dict[Tuple[int, int], int]):
        """
        Update the agent's policy.
        
        Args:
            policy: New policy dictionary
        """
        self.policy = policy
    
    def is_using_policy(self) -> bool:
        """Check if agent has a policy."""
        return self.policy is not None
    
    def get_path_length(self) -> int:
        """Get number of steps taken."""
        return len(self.action_history)
    
    def has_reached_goal(self) -> bool:
        """Check if agent has reached goal."""
        return self.env.is_terminal(self.current_state)
    
    def __repr__(self) -> str:
        """String representation."""
        policy_type = "Policy" if self.policy else "Random"
        return f"Agent({policy_type}, state={self.current_state}, steps={self.get_path_length()})"
