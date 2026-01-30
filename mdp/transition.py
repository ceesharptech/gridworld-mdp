"""
Transition Model Module

This module implements the transition probability function P(s'|s,a) for the MDP.
Supports both deterministic and stochastic transitions with configurable noise.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
from typing import List, Tuple, Dict
from mdp.environment import GridWorld


class TransitionModel:
    """
    Transition probability model for Grid World MDP.
    
    Implements P(s'|s,a): probability of transitioning to state s' 
    given current state s and action a.
    
    Supports:
    - Deterministic transitions (agent moves as intended)
    - Stochastic transitions (agent may slip in perpendicular directions)
    
    Attributes:
        env (GridWorld): The environment
        stochastic (bool): Whether transitions are stochastic
        success_prob (float): Probability of intended action (stochastic mode)
        slip_prob (float): Probability of each perpendicular action
    """
    
    def __init__(
        self,
        env: GridWorld,
        stochastic: bool = False,
        success_prob: float = 0.8
    ):
        """
        Initialize the transition model.
        
        Args:
            env: Grid World environment
            stochastic: If True, use stochastic transitions
            success_prob: Probability of executing intended action (stochastic mode)
                         Remaining probability split between perpendicular actions
        """
        self.env = env
        self.stochastic = stochastic
        self.success_prob = success_prob
        
        # In stochastic mode, remaining probability split between two perpendicular actions
        self.slip_prob = (1.0 - success_prob) / 2.0 if stochastic else 0.0
        
        # Define perpendicular actions for each action
        self._perpendicular_actions = {
            GridWorld.UP: [GridWorld.LEFT, GridWorld.RIGHT],
            GridWorld.DOWN: [GridWorld.LEFT, GridWorld.RIGHT],
            GridWorld.LEFT: [GridWorld.UP, GridWorld.DOWN],
            GridWorld.RIGHT: [GridWorld.UP, GridWorld.DOWN]
        }
    
    def get_transition_prob(
        self,
        state: Tuple[int, int],
        action: int,
        next_state: Tuple[int, int]
    ) -> float:
        """
        Get probability of transitioning to next_state from state with action.
        
        P(s'|s,a)
        
        Args:
            state: Current state (row, col)
            action: Action taken
            next_state: Next state (row, col)
            
        Returns:
            Probability of transition (float between 0 and 1)
        """
        # Terminal states have no outgoing transitions
        if self.env.is_terminal(state):
            return 1.0 if next_state == state else 0.0
        
        if not self.stochastic:
            # Deterministic transitions
            expected_next = self.env.get_next_state(state, action)
            return 1.0 if next_state == expected_next else 0.0
        else:
            # Stochastic transitions
            return self._get_stochastic_prob(state, action, next_state)
    
    def _get_stochastic_prob(
        self,
        state: Tuple[int, int],
        action: int,
        next_state: Tuple[int, int]
    ) -> float:
        """
        Calculate stochastic transition probability.
        
        Agent executes intended action with probability success_prob,
        and slips in perpendicular directions with probability slip_prob each.
        
        Args:
            state: Current state
            action: Intended action
            next_state: Resulting state
            
        Returns:
            Transition probability
        """
        # Get all possible outcomes and their probabilities
        outcomes = self.get_possible_transitions(state, action)
        
        # Find probability for the specific next_state
        for outcome_state, prob in outcomes:
            if outcome_state == next_state:
                return prob
        
        return 0.0
    
    def get_possible_transitions(
        self,
        state: Tuple[int, int],
        action: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get all possible next states and their probabilities for a given state-action pair.
        
        Returns a list of (next_state, probability) tuples.
        
        Args:
            state: Current state (row, col)
            action: Action taken
            
        Returns:
            List of (next_state, probability) tuples
        """
        # Terminal states stay terminal
        if self.env.is_terminal(state):
            return [(state, 1.0)]
        
        if not self.stochastic:
            # Deterministic: only one outcome
            next_state = self.env.get_next_state(state, action)
            return [(next_state, 1.0)]
        
        # Stochastic: multiple possible outcomes
        outcomes = {}
        
        # Intended action
        intended_next = self.env.get_next_state(state, action)
        outcomes[intended_next] = outcomes.get(intended_next, 0.0) + self.success_prob
        
        # Perpendicular slips
        perp_actions = self._perpendicular_actions[action]
        for perp_action in perp_actions:
            perp_next = self.env.get_next_state(state, perp_action)
            outcomes[perp_next] = outcomes.get(perp_next, 0.0) + self.slip_prob
        
        # Convert to list of tuples
        return list(outcomes.items())
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Construct the full transition matrix for all states and actions.
        
        Returns a 3D array T where T[s, a, s'] = P(s'|s,a)
        
        This is computationally expensive and primarily for analysis.
        
        Returns:
            3D numpy array of shape (n_states, n_actions, n_states)
        """
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        
        T = np.zeros((n_states, n_actions, n_states))
        
        state_to_idx = {state: idx for idx, state in enumerate(self.env.states)}
        
        for s_idx, state in enumerate(self.env.states):
            for a_idx, action in enumerate(self.env.actions):
                transitions = self.get_possible_transitions(state, action)
                for next_state, prob in transitions:
                    next_idx = state_to_idx[next_state]
                    T[s_idx, a_idx, next_idx] = prob
        
        return T
    
    def expected_next_state(
        self,
        state: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        """
        Get the most likely next state (maximum probability).
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            Most likely next state
        """
        transitions = self.get_possible_transitions(state, action)
        # Return state with maximum probability
        return max(transitions, key=lambda x: x[1])[0]
    
    def __repr__(self) -> str:
        """String representation."""
        mode = "Stochastic" if self.stochastic else "Deterministic"
        if self.stochastic:
            return f"TransitionModel({mode}, p_success={self.success_prob})"
        return f"TransitionModel({mode})"
