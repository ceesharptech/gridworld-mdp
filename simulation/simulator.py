"""
Simulator Module

This module runs complete episodes in the Grid World environment.
It executes agent policies and tracks performance metrics.

Author: AI Research Project
Date: January 2026
"""

from typing import Dict, Tuple, List, Optional
from mdp.environment import GridWorld
from mdp.transition import TransitionModel
from mdp.rewards import RewardFunction
from simulation.agent import Agent


class Simulator:
    """
    Simulator for running Grid World episodes.
    
    Executes agent navigation from start to goal (or failure) and
    tracks comprehensive metrics including path, rewards, and steps.
    
    Attributes:
        env (GridWorld): The environment
        transition_model (TransitionModel): Transition dynamics
        reward_function (RewardFunction): Reward function
        max_steps (int): Maximum steps before termination
    """
    
    def __init__(
        self,
        env: GridWorld,
        transition_model: TransitionModel,
        reward_function: RewardFunction,
        max_steps: int = 1000
    ):
        """
        Initialize the simulator.
        
        Args:
            env: Grid World environment
            transition_model: Transition probability model
            reward_function: Reward function
            max_steps: Maximum steps before episode terminates
        """
        self.env = env
        self.transition_model = transition_model
        self.reward_function = reward_function
        self.max_steps = max_steps
    
    def run_episode(
        self,
        agent: Agent,
        start_state: Optional[Tuple[int, int]] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Run a complete episode with the agent.
        
        Agent navigates from start state to goal (or until max_steps).
        
        Args:
            agent: Agent with policy
            start_state: Optional custom start state
            verbose: If True, print step-by-step progress
            
        Returns:
            Dictionary containing episode results:
                - path: List of states visited
                - actions: List of actions taken
                - rewards: List of rewards received
                - total_reward: Cumulative reward
                - steps: Number of steps taken
                - success: Whether goal was reached
                - terminated_reason: 'goal' or 'max_steps'
        """
        # Reset agent
        if start_state is None:
            start_state = self.env.start_state
        agent.reset(start_state)
        
        # Initialize tracking
        path = [agent.current_state]
        actions = []
        rewards = []
        total_reward = 0.0
        steps = 0
        success = False
        
        if verbose:
            print(f"Episode Start: {agent.current_state}")
            print("-" * 60)
        
        # Run episode
        while steps < self.max_steps:
            current_state = agent.current_state
            
            # Check if terminal
            if self.env.is_terminal(current_state):
                success = True
                terminated_reason = 'goal'
                if verbose:
                    print(f"Goal reached at step {steps}!")
                break
            
            # Agent selects action
            action = agent.select_action(current_state)
            
            # Execute action (use transition model for stochastic dynamics)
            if self.transition_model.stochastic:
                # Sample next state based on transition probabilities
                transitions = self.transition_model.get_possible_transitions(
                    current_state, action
                )
                import random
                states, probs = zip(*transitions)
                next_state = random.choices(states, weights=probs, k=1)[0]
            else:
                # Deterministic transition
                next_state = self.env.get_next_state(current_state, action)
            
            # Calculate reward
            reward = self.reward_function.get_reward(current_state, action, next_state)
            
            # Update agent
            agent.current_state = next_state
            
            # Record
            actions.append(action)
            path.append(next_state)
            rewards.append(reward)
            total_reward += reward
            steps += 1
            
            if verbose:
                action_symbol = self.env.get_action_name(action)
                print(f"Step {steps}: {current_state} --{action_symbol}--> {next_state} "
                      f"(R={reward:.1f}, Total={total_reward:.1f})")
        
        # Check termination reason
        if steps >= self.max_steps:
            terminated_reason = 'max_steps'
            if verbose:
                print(f"Terminated: Maximum steps ({self.max_steps}) reached")
        
        if verbose:
            print("-" * 60)
            print(f"Episode End: Success={success}, Steps={steps}, "
                  f"Total Reward={total_reward:.2f}")
        
        return {
            'path': path,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'terminated_reason': terminated_reason,
            'final_state': agent.current_state
        }
    
    def run_multiple_episodes(
        self,
        agent: Agent,
        num_episodes: int = 10,
        start_state: Optional[Tuple[int, int]] = None
    ) -> List[Dict]:
        """
        Run multiple episodes and collect results.
        
        Useful for evaluating stochastic policies or environments.
        
        Args:
            agent: Agent with policy
            num_episodes: Number of episodes to run
            start_state: Optional custom start state
            
        Returns:
            List of episode result dictionaries
        """
        results = []
        
        for i in range(num_episodes):
            episode_result = self.run_episode(agent, start_state, verbose=False)
            episode_result['episode_number'] = i + 1
            results.append(episode_result)
        
        return results
    
    def compare_policies(
        self,
        policies: Dict[str, Dict],
        num_episodes: int = 1,
        start_state: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Compare multiple policies.
        
        Args:
            policies: Dictionary mapping policy names to policy dictionaries
            num_episodes: Episodes per policy (for stochastic environments)
            start_state: Optional custom start state
            
        Returns:
            Dictionary comparing policy performance
        """
        comparison = {}
        
        for policy_name, policy in policies.items():
            agent = Agent(self.env, policy)
            episodes = self.run_multiple_episodes(agent, num_episodes, start_state)
            
            # Aggregate statistics
            total_rewards = [ep['total_reward'] for ep in episodes]
            step_counts = [ep['steps'] for ep in episodes]
            success_count = sum(1 for ep in episodes if ep['success'])
            
            comparison[policy_name] = {
                'episodes': episodes,
                'avg_reward': sum(total_rewards) / len(total_rewards),
                'avg_steps': sum(step_counts) / len(step_counts),
                'success_rate': success_count / num_episodes,
                'best_reward': max(total_rewards),
                'worst_reward': min(total_rewards)
            }
        
        return comparison
    
    def evaluate_policy(
        self,
        policy: Dict[Tuple[int, int], int],
        num_episodes: int = 10
    ) -> Dict:
        """
        Evaluate a single policy over multiple episodes.
        
        Args:
            policy: Policy dictionary
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with evaluation metrics
        """
        agent = Agent(self.env, policy)
        episodes = self.run_multiple_episodes(agent, num_episodes)
        
        # Calculate statistics
        total_rewards = [ep['total_reward'] for ep in episodes]
        step_counts = [ep['steps'] for ep in episodes]
        success_count = sum(1 for ep in episodes if ep['success'])
        
        return {
            'num_episodes': num_episodes,
            'avg_reward': sum(total_rewards) / num_episodes,
            'std_reward': (sum((r - sum(total_rewards)/num_episodes)**2 
                              for r in total_rewards) / num_episodes) ** 0.5,
            'avg_steps': sum(step_counts) / num_episodes,
            'std_steps': (sum((s - sum(step_counts)/num_episodes)**2 
                             for s in step_counts) / num_episodes) ** 0.5,
            'success_rate': success_count / num_episodes,
            'min_reward': min(total_rewards),
            'max_reward': max(total_rewards),
            'min_steps': min(step_counts),
            'max_steps': max(step_counts)
        }
    
    def get_optimal_path_length(self) -> int:
        """
        Calculate Manhattan distance from start to goal.
        
        This is the theoretical minimum steps in a grid without obstacles.
        
        Returns:
            Minimum possible steps
        """
        start = self.env.start_state
        goal = self.env.goal_state
        return abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Simulator(env={self.env}, max_steps={self.max_steps})"
