"""
Metrics Module

This module computes evaluation metrics for MDP solutions.
Tracks performance, convergence, and solution quality.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Tuple


class Metrics:
    """
    Evaluation metrics for MDP solutions.
    
    Computes various metrics including:
    - Episode performance (reward, steps, success)
    - Convergence metrics (iterations, delta)
    - Policy quality metrics
    - Comparative metrics
    """
    
    @staticmethod
    def compute_episode_metrics(episode_result: Dict) -> Dict:
        """
        Compute metrics for a single episode.
        
        Args:
            episode_result: Dictionary from Simulator.run_episode()
            
        Returns:
            Dictionary with computed metrics
        """
        metrics = {
            'total_reward': episode_result['total_reward'],
            'num_steps': episode_result['steps'],
            'success': episode_result['success'],
            'path_length': len(episode_result['path']),
            'unique_states_visited': len(set(episode_result['path'])),
        }
        
        # Reward statistics
        if episode_result['rewards']:
            metrics['avg_reward_per_step'] = (
                episode_result['total_reward'] / len(episode_result['rewards'])
            )
            metrics['min_reward'] = min(episode_result['rewards'])
            metrics['max_reward'] = max(episode_result['rewards'])
        else:
            metrics['avg_reward_per_step'] = 0.0
            metrics['min_reward'] = 0.0
            metrics['max_reward'] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_multi_episode_metrics(episodes: List[Dict]) -> Dict:
        """
        Compute aggregate metrics across multiple episodes.
        
        Args:
            episodes: List of episode result dictionaries
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not episodes:
            return {}
        
        total_rewards = [ep['total_reward'] for ep in episodes]
        step_counts = [ep['steps'] for ep in episodes]
        successes = [ep['success'] for ep in episodes]
        
        metrics = {
            'num_episodes': len(episodes),
            'success_rate': sum(successes) / len(successes),
            
            # Reward statistics
            'avg_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'min_total_reward': np.min(total_rewards),
            'max_total_reward': np.max(total_rewards),
            
            # Step statistics
            'avg_steps': np.mean(step_counts),
            'std_steps': np.std(step_counts),
            'min_steps': np.min(step_counts),
            'max_steps': np.max(step_counts),
        }
        
        # Successful episodes only
        successful_episodes = [ep for ep in episodes if ep['success']]
        if successful_episodes:
            successful_rewards = [ep['total_reward'] for ep in successful_episodes]
            successful_steps = [ep['steps'] for ep in successful_episodes]
            
            metrics['avg_reward_when_successful'] = np.mean(successful_rewards)
            metrics['avg_steps_when_successful'] = np.mean(successful_steps)
        
        return metrics
    
    @staticmethod
    def compute_convergence_metrics(convergence_history: List[float]) -> Dict:
        """
        Compute metrics about Value Iteration convergence.
        
        Args:
            convergence_history: List of delta values per iteration
            
        Returns:
            Dictionary with convergence metrics
        """
        if not convergence_history:
            return {}
        
        metrics = {
            'num_iterations': len(convergence_history),
            'final_delta': convergence_history[-1],
            'initial_delta': convergence_history[0],
            'avg_delta': np.mean(convergence_history),
            'convergence_rate': None
        }
        
        # Estimate convergence rate (exponential decay rate)
        if len(convergence_history) > 1:
            # Simple linear regression on log(delta)
            log_deltas = [np.log(max(d, 1e-10)) for d in convergence_history]
            iterations = list(range(len(convergence_history)))
            
            # Calculate slope (convergence rate)
            n = len(iterations)
            sum_x = sum(iterations)
            sum_y = sum(log_deltas)
            sum_xy = sum(x * y for x, y in zip(iterations, log_deltas))
            sum_x2 = sum(x * x for x in iterations)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                metrics['convergence_rate'] = slope
        
        return metrics
    
    @staticmethod
    def compute_policy_metrics(
        policy: Dict[Tuple[int, int], int],
        env
    ) -> Dict:
        """
        Compute metrics about the policy itself.
        
        Args:
            policy: Policy dictionary mapping states to actions
            env: GridWorld environment
            
        Returns:
            Dictionary with policy metrics
        """
        action_counts = {}
        terminal_count = 0
        
        for state, action in policy.items():
            if action is None or env.is_terminal(state):
                terminal_count += 1
            else:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        non_terminal = len(policy) - terminal_count
        
        metrics = {
            'total_states': len(policy),
            'terminal_states': terminal_count,
            'non_terminal_states': non_terminal,
            'action_distribution': action_counts,
        }
        
        # Action entropy (measure of policy diversity)
        if non_terminal > 0:
            action_probs = [count / non_terminal for count in action_counts.values()]
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in action_probs)
            metrics['action_entropy'] = entropy
            metrics['action_entropy_normalized'] = entropy / np.log2(len(env.actions))
        
        return metrics
    
    @staticmethod
    def compute_value_function_metrics(
        value_function: Dict[Tuple[int, int], float],
        env
    ) -> Dict:
        """
        Compute metrics about the value function.
        
        Args:
            value_function: Dictionary mapping states to values
            env: GridWorld environment
            
        Returns:
            Dictionary with value function metrics
        """
        non_terminal_values = [
            value_function[state] for state in env.states 
            if not env.is_terminal(state)
        ]
        
        if not non_terminal_values:
            return {}
        
        metrics = {
            'avg_state_value': np.mean(non_terminal_values),
            'std_state_value': np.std(non_terminal_values),
            'min_state_value': np.min(non_terminal_values),
            'max_state_value': np.max(non_terminal_values),
            'value_range': np.max(non_terminal_values) - np.min(non_terminal_values)
        }
        
        return metrics
    
    @staticmethod
    def compare_policies(
        policy_results: Dict[str, Dict]
    ) -> Dict:
        """
        Compare multiple policies.
        
        Args:
            policy_results: Dictionary mapping policy names to their results
                           (from Simulator.compare_policies())
            
        Returns:
            Comparison metrics and rankings
        """
        if not policy_results:
            return {}
        
        # Rank policies by different criteria
        rankings = {
            'by_avg_reward': sorted(
                policy_results.items(),
                key=lambda x: x[1]['avg_reward'],
                reverse=True
            ),
            'by_avg_steps': sorted(
                policy_results.items(),
                key=lambda x: x[1]['avg_steps']
            ),
            'by_success_rate': sorted(
                policy_results.items(),
                key=lambda x: x[1]['success_rate'],
                reverse=True
            )
        }
        
        comparison = {
            'num_policies': len(policy_results),
            'rankings': rankings,
            'policy_names': list(policy_results.keys())
        }
        
        # Best overall (by reward)
        if rankings['by_avg_reward']:
            best_policy_name = rankings['by_avg_reward'][0][0]
            comparison['best_policy'] = best_policy_name
            comparison['best_avg_reward'] = policy_results[best_policy_name]['avg_reward']
        
        return comparison
    
    @staticmethod
    def compute_efficiency_metric(
        actual_steps: int,
        optimal_steps: int
    ) -> float:
        """
        Compute path efficiency as ratio of optimal to actual steps.
        
        Args:
            actual_steps: Steps taken by policy
            optimal_steps: Theoretical minimum steps
            
        Returns:
            Efficiency ratio (1.0 = optimal, lower = less efficient)
        """
        if actual_steps <= 0:
            return 0.0
        return optimal_steps / actual_steps
    
    @staticmethod
    def format_metrics_report(metrics: Dict, title: str = "Metrics Report") -> str:
        """
        Format metrics into a readable string report.
        
        Args:
            metrics: Dictionary of metrics
            title: Report title
            
        Returns:
            Formatted string
        """
        lines = [
            "=" * 60,
            title,
            "=" * 60
        ]
        
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key:.<40} {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        lines.append(f"  {sub_key:.<38} {sub_value:.4f}")
                    else:
                        lines.append(f"  {sub_key:.<38} {sub_value}")
            else:
                lines.append(f"{key:.<40} {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
