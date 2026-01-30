"""
Experiments Module

This module provides functionality for running controlled experiments
and parameter sweeps for academic analysis.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from mdp.environment import GridWorld
from mdp.transition import TransitionModel
from mdp.rewards import RewardFunction
from algorithms.value_iteration import ValueIteration
from algorithms.policy_extraction import PolicyExtractor
from simulation.simulator import Simulator
from evaluation.metrics import Metrics


class Experiments:
    """
    Experiment runner for systematic MDP analysis.
    
    Supports:
    - Parameter sweeps (gamma, step penalty, grid size)
    - Stochastic vs deterministic comparison
    - Multiple random seeds
    - Statistical analysis
    """
    
    def __init__(self, base_env: GridWorld = None):
        """
        Initialize experiment runner.
        
        Args:
            base_env: Optional base environment for experiments
        """
        self.base_env = base_env
        self.results = {}
    
    def gamma_sweep(
        self,
        env: GridWorld,
        gamma_values: List[float],
        num_trials: int = 1,
        verbose: bool = False
    ) -> Dict:
        """
        Sweep across different discount factor values.
        
        Args:
            env: Grid World environment
            gamma_values: List of gamma values to test
            num_trials: Number of trials per gamma (for stochastic)
            verbose: Print progress
            
        Returns:
            Dictionary with results for each gamma
        """
        results = {}
        
        if verbose:
            print(f"Running gamma sweep: {gamma_values}")
            print("=" * 60)
        
        for gamma in gamma_values:
            if verbose:
                print(f"\nTesting γ = {gamma}")
            
            # Setup
            transition_model = TransitionModel(env, stochastic=False)
            reward_function = RewardFunction(env)
            
            # Solve MDP
            vi = ValueIteration(env, transition_model, reward_function, gamma=gamma)
            vi_result = vi.solve(verbose=False)
            
            # Extract policy
            extractor = PolicyExtractor(env, transition_model, reward_function, gamma)
            policy = extractor.extract_policy(vi_result['value_function'])
            
            # Evaluate
            simulator = Simulator(env, transition_model, reward_function)
            eval_result = simulator.evaluate_policy(policy, num_episodes=num_trials)
            
            # Store results
            results[gamma] = {
                'gamma': gamma,
                'iterations': vi_result['iterations'],
                'converged': vi_result['converged'],
                'avg_reward': eval_result['avg_reward'],
                'avg_steps': eval_result['avg_steps'],
                'success_rate': eval_result['success_rate'],
                'value_function': vi_result['value_function'],
                'policy': policy
            }
            
            if verbose:
                print(f"  Iterations: {vi_result['iterations']}")
                print(f"  Avg Reward: {eval_result['avg_reward']:.2f}")
                print(f"  Avg Steps: {eval_result['avg_steps']:.1f}")
        
        return results
    
    def stochastic_comparison(
        self,
        env: GridWorld,
        success_probs: List[float],
        num_episodes: int = 10,
        verbose: bool = False
    ) -> Dict:
        """
        Compare deterministic vs stochastic transitions.
        
        Args:
            env: Grid World environment
            success_probs: List of success probabilities for stochastic mode
            num_episodes: Episodes per configuration
            verbose: Print progress
            
        Returns:
            Comparison results
        """
        results = {}
        
        if verbose:
            print("Comparing Deterministic vs Stochastic Transitions")
            print("=" * 60)
        
        # Deterministic baseline
        if verbose:
            print("\nDeterministic (p=1.0)")
        
        transition_model = TransitionModel(env, stochastic=False)
        reward_function = RewardFunction(env)
        
        vi = ValueIteration(env, transition_model, reward_function, gamma=0.9)
        vi_result = vi.solve(verbose=False)
        
        extractor = PolicyExtractor(env, transition_model, reward_function, 0.9)
        policy = extractor.extract_policy(vi_result['value_function'])
        
        simulator = Simulator(env, transition_model, reward_function)
        eval_result = simulator.evaluate_policy(policy, num_episodes)
        
        results['deterministic'] = {
            'success_prob': 1.0,
            'stochastic': False,
            'iterations': vi_result['iterations'],
            'avg_reward': eval_result['avg_reward'],
            'avg_steps': eval_result['avg_steps'],
            'success_rate': eval_result['success_rate']
        }
        
        if verbose:
            print(f"  Avg Reward: {eval_result['avg_reward']:.2f}")
            print(f"  Success Rate: {eval_result['success_rate']:.2%}")
        
        # Stochastic variations
        for success_prob in success_probs:
            if verbose:
                print(f"\nStochastic (p={success_prob})")
            
            transition_model = TransitionModel(env, stochastic=True, 
                                             success_prob=success_prob)
            
            vi = ValueIteration(env, transition_model, reward_function, gamma=0.9)
            vi_result = vi.solve(verbose=False)
            
            extractor = PolicyExtractor(env, transition_model, reward_function, 0.9)
            policy = extractor.extract_policy(vi_result['value_function'])
            
            simulator = Simulator(env, transition_model, reward_function)
            eval_result = simulator.evaluate_policy(policy, num_episodes)
            
            results[f'stochastic_{success_prob}'] = {
                'success_prob': success_prob,
                'stochastic': True,
                'iterations': vi_result['iterations'],
                'avg_reward': eval_result['avg_reward'],
                'std_reward': eval_result['std_reward'],
                'avg_steps': eval_result['avg_steps'],
                'success_rate': eval_result['success_rate']
            }
            
            if verbose:
                print(f"  Avg Reward: {eval_result['avg_reward']:.2f} "
                      f"(±{eval_result['std_reward']:.2f})")
                print(f"  Success Rate: {eval_result['success_rate']:.2%}")
        
        return results
    
    def grid_size_experiment(
        self,
        grid_sizes: List[Tuple[int, int]],
        verbose: bool = False
    ) -> Dict:
        """
        Test performance across different grid sizes.
        
        Args:
            grid_sizes: List of (rows, cols) tuples
            verbose: Print progress
            
        Returns:
            Results for each grid size
        """
        results = {}
        
        if verbose:
            print("Grid Size Scaling Experiment")
            print("=" * 60)
        
        for size in grid_sizes:
            rows, cols = size
            
            if verbose:
                print(f"\nGrid Size: {rows}x{cols}")
            
            # Create environment with scaled goal
            env = GridWorld(
                grid_size=size,
                start_state=(0, 0),
                goal_state=(rows-1, cols-1)
            )
            
            transition_model = TransitionModel(env, stochastic=False)
            reward_function = RewardFunction(env)
            
            # Solve
            vi = ValueIteration(env, transition_model, reward_function, gamma=0.9)
            vi_result = vi.solve(verbose=False)
            
            # Extract and evaluate
            extractor = PolicyExtractor(env, transition_model, reward_function, 0.9)
            policy = extractor.extract_policy(vi_result['value_function'])
            
            simulator = Simulator(env, transition_model, reward_function)
            eval_result = simulator.evaluate_policy(policy, num_episodes=1)
            
            # Calculate metrics
            optimal_steps = simulator.get_optimal_path_length()
            efficiency = Metrics.compute_efficiency_metric(
                eval_result['avg_steps'], optimal_steps
            )
            
            results[f'{rows}x{cols}'] = {
                'grid_size': size,
                'num_states': env.n_states,
                'iterations': vi_result['iterations'],
                'avg_steps': eval_result['avg_steps'],
                'optimal_steps': optimal_steps,
                'efficiency': efficiency,
                'avg_reward': eval_result['avg_reward']
            }
            
            if verbose:
                print(f"  States: {env.n_states}")
                print(f"  Iterations: {vi_result['iterations']}")
                print(f"  Steps: {eval_result['avg_steps']:.1f} "
                      f"(optimal: {optimal_steps})")
                print(f"  Efficiency: {efficiency:.2%}")
        
        return results
    
    def reward_structure_experiment(
        self,
        env: GridWorld,
        reward_configs: List[Dict],
        verbose: bool = False
    ) -> Dict:
        """
        Test different reward function configurations.
        
        Args:
            env: Grid World environment
            reward_configs: List of reward configuration dictionaries
            verbose: Print progress
            
        Returns:
            Results for each configuration
        """
        results = {}
        
        if verbose:
            print("Reward Structure Experiment")
            print("=" * 60)
        
        transition_model = TransitionModel(env, stochastic=False)
        
        for i, config in enumerate(reward_configs):
            config_name = config.get('name', f'config_{i}')
            
            if verbose:
                print(f"\n{config_name}")
                print(f"  Goal: {config.get('goal_reward', 100)}")
                print(f"  Step: {config.get('step_penalty', -1)}")
            
            # Create reward function
            reward_function = RewardFunction(
                env,
                goal_reward=config.get('goal_reward', 100),
                step_penalty=config.get('step_penalty', -1),
                obstacle_penalty=config.get('obstacle_penalty', -10)
            )
            
            # Solve
            vi = ValueIteration(env, transition_model, reward_function, gamma=0.9)
            vi_result = vi.solve(verbose=False)
            
            # Evaluate
            extractor = PolicyExtractor(env, transition_model, reward_function, 0.9)
            policy = extractor.extract_policy(vi_result['value_function'])
            
            simulator = Simulator(env, transition_model, reward_function)
            eval_result = simulator.evaluate_policy(policy, num_episodes=1)
            
            results[config_name] = {
                'config': config,
                'iterations': vi_result['iterations'],
                'avg_reward': eval_result['avg_reward'],
                'avg_steps': eval_result['avg_steps'],
                'success_rate': eval_result['success_rate']
            }
            
            if verbose:
                print(f"  Result - Steps: {eval_result['avg_steps']:.1f}, "
                      f"Reward: {eval_result['avg_reward']:.2f}")
        
        return results
    
    def convergence_analysis(
        self,
        env: GridWorld,
        theta_values: List[float],
        verbose: bool = False
    ) -> Dict:
        """
        Analyze convergence behavior for different thresholds.
        
        Args:
            env: Grid World environment
            theta_values: List of convergence thresholds
            verbose: Print progress
            
        Returns:
            Convergence analysis results
        """
        results = {}
        
        transition_model = TransitionModel(env, stochastic=False)
        reward_function = RewardFunction(env)
        
        if verbose:
            print("Convergence Threshold Analysis")
            print("=" * 60)
        
        for theta in theta_values:
            if verbose:
                print(f"\nθ = {theta}")
            
            vi = ValueIteration(env, transition_model, reward_function, 
                              gamma=0.9, theta=theta)
            vi_result = vi.solve(verbose=False)
            
            conv_metrics = Metrics.compute_convergence_metrics(
                vi_result['convergence_history']
            )
            
            results[theta] = {
                'theta': theta,
                'iterations': vi_result['iterations'],
                'converged': vi_result['converged'],
                'final_delta': vi_result['convergence_history'][-1],
                'convergence_metrics': conv_metrics
            }
            
            if verbose:
                print(f"  Iterations: {vi_result['iterations']}")
                print(f"  Final δ: {vi_result['convergence_history'][-1]:.2e}")
        
        return results
    
    def save_results(self, results: Dict, name: str):
        """Store experiment results."""
        self.results[name] = results
    
    def get_results(self, name: str) -> Dict:
        """Retrieve stored results."""
        return self.results.get(name, {})
