"""
Plots Module

This module provides plotting functions for analysis and evaluation,
including convergence plots, reward curves, and comparison charts.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple


class Plotter:
    """
    Plotting utilities for MDP analysis.
    
    Provides methods to create:
    - Convergence plots
    - Reward vs iteration curves
    - Policy comparison charts
    - Parameter sweep visualizations
    """
    
    @staticmethod
    def plot_convergence(
        convergence_history: List[float],
        title: str = "Value Iteration Convergence",
        ax: Optional[plt.Axes] = None,
        log_scale: bool = True
    ) -> plt.Axes:
        """
        Plot convergence history of Value Iteration.
        
        Args:
            convergence_history: List of delta values per iteration
            title: Plot title
            ax: Matplotlib axes
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(convergence_history) + 1)
        
        ax.plot(iterations, convergence_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Maximum Value Change (δ)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add convergence info
        final_delta = convergence_history[-1]
        ax.axhline(y=final_delta, color='r', linestyle='--', alpha=0.5,
                  label=f'Final δ = {final_delta:.2e}')
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_episode_rewards(
        rewards: List[float],
        title: str = "Rewards per Step",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot rewards obtained at each step of an episode.
        
        Args:
            rewards: List of rewards per step
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = range(1, len(rewards) + 1)
        cumulative_rewards = np.cumsum(rewards)
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        # Plot immediate rewards
        ax.bar(steps, rewards, alpha=0.6, color='skyblue', label='Immediate Reward')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Immediate Reward', fontsize=12, color='skyblue')
        ax.tick_params(axis='y', labelcolor='skyblue')
        
        # Plot cumulative rewards
        ax2.plot(steps, cumulative_rewards, 'r-', linewidth=2, 
                label='Cumulative Reward')
        ax2.set_ylabel('Cumulative Reward', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        return ax
    
    @staticmethod
    def plot_gamma_sweep(
        results: Dict[float, Dict],
        metric: str = 'avg_reward',
        title: str = "Discount Factor Sweep",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot results of gamma parameter sweep.
        
        Args:
            results: Results from Experiments.gamma_sweep()
            metric: Metric to plot ('avg_reward', 'avg_steps', 'iterations')
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        gammas = sorted(results.keys())
        values = [results[g][metric] for g in gammas]
        
        ax.plot(gammas, values, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Discount Factor (γ)', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight optimal gamma
        if metric in ['avg_reward', 'success_rate']:
            optimal_idx = np.argmax(values)
        else:  # For steps or iterations, lower is better
            optimal_idx = np.argmin(values)
        
        optimal_gamma = gammas[optimal_idx]
        optimal_value = values[optimal_idx]
        ax.plot(optimal_gamma, optimal_value, 'r*', markersize=20,
               label=f'Optimal: γ={optimal_gamma:.2f}')
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_policy_comparison(
        comparison: Dict[str, Dict],
        metric: str = 'avg_reward',
        title: str = "Policy Comparison",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Create bar chart comparing multiple policies.
        
        Args:
            comparison: Results from Simulator.compare_policies()
            metric: Metric to compare
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        policy_names = list(comparison.keys())
        values = [comparison[name][metric] for name in policy_names]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(policy_names)))
        bars = ax.bar(policy_names, values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return ax
    
    @staticmethod
    def plot_stochastic_comparison(
        results: Dict[str, Dict],
        title: str = "Deterministic vs Stochastic Performance",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Compare deterministic and stochastic environments.
        
        Args:
            results: Results from Experiments.stochastic_comparison()
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        configs = []
        success_probs = []
        avg_rewards = []
        std_rewards = []
        
        for config_name, data in sorted(results.items()):
            configs.append(config_name)
            success_probs.append(data['success_prob'])
            avg_rewards.append(data['avg_reward'])
            std_rewards.append(data.get('std_reward', 0))
        
        x = np.arange(len(configs))
        
        # Plot with error bars
        ax.errorbar(x, avg_rewards, yerr=std_rewards, fmt='o-', 
                   linewidth=2, markersize=8, capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'p={p:.1f}' for p in success_probs], rotation=45)
        ax.set_xlabel('Success Probability', fontsize=12)
        ax.set_ylabel('Average Total Reward', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return ax
    
    @staticmethod
    def plot_multi_metric_comparison(
        results: Dict[str, Dict],
        metrics: List[str] = ['avg_reward', 'avg_steps', 'success_rate'],
        title: str = "Multi-Metric Comparison"
    ) -> plt.Figure:
        """
        Create multi-panel comparison of different metrics.
        
        Args:
            results: Results dictionary
            metrics: List of metrics to plot
            title: Overall title
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        policy_names = list(results.keys())
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [results[name][metric] for name in policy_names]
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(policy_names)))
            bars = ax.bar(policy_names, values, color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_grid_size_scaling(
        results: Dict[str, Dict],
        title: str = "Grid Size Scaling Analysis",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot how performance scales with grid size.
        
        Args:
            results: Results from Experiments.grid_size_experiment()
            title: Plot title
            ax: Matplotlib axes
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by number of states
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['num_states'])
        
        grid_sizes = [item[0] for item in sorted_results]
        num_states = [item[1]['num_states'] for item in sorted_results]
        iterations = [item[1]['iterations'] for item in sorted_results]
        
        ax2 = ax.twinx()
        
        # Plot iterations vs states
        ax.plot(num_states, iterations, 'bo-', linewidth=2, markersize=8,
               label='Iterations')
        ax.set_xlabel('Number of States', fontsize=12)
        ax.set_ylabel('Iterations to Converge', fontsize=12, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Plot efficiency
        efficiencies = [item[1]['efficiency'] for item in sorted_results]
        ax2.plot(num_states, efficiencies, 'ro-', linewidth=2, markersize=8,
                label='Efficiency')
        ax2.set_ylabel('Path Efficiency', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add grid size labels
        for i, (size, states) in enumerate(zip(grid_sizes, num_states)):
            if i % 2 == 0:  # Label every other point to avoid crowding
                ax.annotate(size, (states, iterations[i]),
                          textcoords="offset points", xytext=(0,10),
                          ha='center', fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        return ax
    
    @staticmethod
    def save_all_plots(
        figures: List[plt.Figure],
        filenames: List[str],
        output_dir: str = "."
    ):
        """
        Save multiple figures to files.
        
        Args:
            figures: List of matplotlib figures
            filenames: List of output filenames
            output_dir: Output directory
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for fig, filename in zip(figures, filenames):
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
