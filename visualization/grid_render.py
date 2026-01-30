"""
Grid Rendering Module

This module provides visualization functions for the Grid World,
including value functions, policies, and agent paths.

Author: AI Research Project
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
from mdp.environment import GridWorld


class GridRenderer:
    """
    Renderer for Grid World visualizations.
    
    Provides methods to visualize:
    - Grid structure (start, goal, obstacles)
    - Value function as heatmap
    - Policy arrows
    - Agent trajectories
    """
    
    def __init__(self, env: GridWorld, figsize: Tuple[int, int] = (10, 10)):
        """
        Initialize grid renderer.
        
        Args:
            env: Grid World environment
            figsize: Figure size for matplotlib plots
        """
        self.env = env
        self.figsize = figsize
        
        # Define colors
        self.colors = {
            'start': '#4CAF50',      # Green
            'goal': '#FFC107',       # Amber
            'obstacle': '#424242',   # Dark grey
            'path': '#2196F3',       # Blue
            'empty': '#FFFFFF',      # White
            'grid_line': '#BDBDBD'   # Light grey
        }
    
    def render_grid_structure(
        self,
        ax: Optional[plt.Axes] = None,
        show_coordinates: bool = False
    ) -> plt.Axes:
        """
        Render basic grid structure with start, goal, and obstacles.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            show_coordinates: Whether to display (row, col) coordinates
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        rows, cols = self.env.grid_size
        
        # Draw grid cells
        for row in range(rows):
            for col in range(cols):
                state = (row, col)
                
                # Determine cell color
                if state == self.env.start_state:
                    color = self.colors['start']
                    label = 'S'
                elif state == self.env.goal_state:
                    color = self.colors['goal']
                    label = 'G'
                elif state in self.env.obstacles:
                    color = self.colors['obstacle']
                    label = 'X'
                else:
                    color = self.colors['empty']
                    label = ''
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (col, rows - row - 1), 1, 1,
                    linewidth=1,
                    edgecolor=self.colors['grid_line'],
                    facecolor=color
                )
                ax.add_patch(rect)
                
                # Add label
                if label:
                    ax.text(
                        col + 0.5, rows - row - 0.5, label,
                        ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if state in self.env.obstacles else 'black'
                    )
                
                # Add coordinates
                if show_coordinates:
                    ax.text(
                        col + 0.1, rows - row - 0.9, f"({row},{col})",
                        ha='left', va='top',
                        fontsize=6, color='gray'
                    )
        
        # Configure axes
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(cols + 1))
        ax.set_yticks(range(rows + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color=self.colors['grid_line'], linewidth=0.5)
        
        return ax
    
    def render_value_function(
        self,
        value_function: Dict[Tuple[int, int], float],
        ax: Optional[plt.Axes] = None,
        show_values: bool = True,
        cmap: str = 'RdYlGn'
    ) -> plt.Axes:
        """
        Render value function as heatmap.
        
        Args:
            value_function: Dictionary mapping states to values
            ax: Matplotlib axes
            show_values: Whether to display numeric values
            cmap: Colormap name
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        rows, cols = self.env.grid_size
        
        # Create value grid
        value_grid = np.zeros((rows, cols))
        for state, value in value_function.items():
            row, col = state
            value_grid[row, col] = value
        
        # Set obstacles to NaN for proper visualization
        for obs in self.env.obstacles:
            row, col = obs
            value_grid[row, col] = np.nan
        
        # Plot heatmap
        im = ax.imshow(
            value_grid,
            cmap=cmap,
            aspect='equal',
            interpolation='nearest'
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='State Value V(s)')
        
        # Add value text
        if show_values:
            for row in range(rows):
                for col in range(cols):
                    state = (row, col)
                    if state not in self.env.obstacles:
                        value = value_function.get(state, 0.0)
                        ax.text(
                            col, row, f'{value:.1f}',
                            ha='center', va='center',
                            fontsize=8,
                            color='black' if abs(value) < np.nanmax(np.abs(value_grid))/2 else 'white'
                        )
        
        # Mark start and goal
        start_row, start_col = self.env.start_state
        goal_row, goal_col = self.env.goal_state
        
        ax.scatter(start_col, start_row, marker='s', s=200, 
                  c='green', edgecolors='black', linewidths=2, label='Start')
        ax.scatter(goal_col, goal_row, marker='*', s=300,
                  c='gold', edgecolors='black', linewidths=2, label='Goal')
        
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols))
        ax.set_yticklabels(range(rows))
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
        ax.set_title('Value Function V(s)', fontsize=14, fontweight='bold')
        
        return ax
    
    def render_policy(
        self,
        policy: Dict[Tuple[int, int], int],
        ax: Optional[plt.Axes] = None,
        arrow_scale: float = 0.4
    ) -> plt.Axes:
        """
        Render policy as arrows on grid.
        
        Args:
            policy: Dictionary mapping states to actions
            ax: Matplotlib axes
            arrow_scale: Scale factor for arrow size
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # First render grid structure
        self.render_grid_structure(ax)
        
        rows, cols = self.env.grid_size
        
        # Arrow directions
        arrow_deltas = {
            self.env.UP: (0, arrow_scale),
            self.env.DOWN: (0, -arrow_scale),
            self.env.LEFT: (-arrow_scale, 0),
            self.env.RIGHT: (arrow_scale, 0)
        }
        
        # Draw policy arrows
        for state, action in policy.items():
            if action is None or self.env.is_terminal(state):
                continue
            
            row, col = state
            if state in self.env.obstacles:
                continue
            
            # Calculate arrow position (center of cell)
            x = col + 0.5
            y = rows - row - 0.5
            
            # Get arrow direction
            dx, dy = arrow_deltas[action]
            
            # Draw arrow
            ax.arrow(
                x, y, dx, dy,
                head_width=0.15,
                head_length=0.15,
                fc='black',
                ec='black',
                linewidth=2
            )
        
        ax.set_title('Policy π(s)', fontsize=14, fontweight='bold')
        
        return ax
    
    def render_path(
        self,
        path: List[Tuple[int, int]],
        ax: Optional[plt.Axes] = None,
        show_grid: bool = True
    ) -> plt.Axes:
        """
        Render agent path through grid.
        
        Args:
            path: List of states visited
            ax: Matplotlib axes
            show_grid: Whether to show grid structure
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if show_grid:
            self.render_grid_structure(ax)
        
        rows, cols = self.env.grid_size
        
        if len(path) > 1:
            # Convert path to plot coordinates
            path_x = [col + 0.5 for row, col in path]
            path_y = [rows - row - 0.5 for row, col in path]
            
            # Plot path
            ax.plot(
                path_x, path_y,
                color=self.colors['path'],
                linewidth=3,
                marker='o',
                markersize=8,
                label=f'Path ({len(path)-1} steps)'
            )
            
            # Mark start and end
            ax.plot(path_x[0], path_y[0], 'go', markersize=15, label='Start')
            ax.plot(path_x[-1], path_y[-1], 'r*', markersize=20, label='End')
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_title('Agent Path', fontsize=14, fontweight='bold')
        
        return ax
    
    def render_combined(
        self,
        value_function: Dict[Tuple[int, int], float],
        policy: Dict[Tuple[int, int], int],
        path: Optional[List[Tuple[int, int]]] = None
    ) -> plt.Figure:
        """
        Create combined visualization with multiple subplots.
        
        Args:
            value_function: Value function
            policy: Policy
            path: Optional agent path
            
        Returns:
            Matplotlib figure
        """
        if path is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            self.render_value_function(value_function, ax=axes[0], show_values=False)
            self.render_policy(policy, ax=axes[1])
            self.render_path(path, ax=axes[2])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            self.render_value_function(value_function, ax=axes[0])
            self.render_policy(policy, ax=axes[1])
        
        fig.suptitle('Grid World MDP Solution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_text_grid(
        self,
        policy: Optional[Dict[Tuple[int, int], int]] = None,
        value_function: Optional[Dict[Tuple[int, int], float]] = None
    ) -> str:
        """
        Create ASCII text representation of grid.
        
        Args:
            policy: Optional policy to display
            value_function: Optional values to display
            
        Returns:
            String representation
        """
        rows, cols = self.env.grid_size
        lines = []
        
        # Header
        lines.append("Grid World:")
        lines.append("─" * (cols * 4 + 1))
        
        for row in range(rows):
            line = "|"
            for col in range(cols):
                state = (row, col)
                
                if state == self.env.start_state:
                    cell = " S "
                elif state == self.env.goal_state:
                    cell = " G "
                elif state in self.env.obstacles:
                    cell = " X "
                elif policy and state in policy and policy[state] is not None:
                    action = policy[state]
                    cell = f" {self.env.get_action_name(action)} "
                elif value_function and state in value_function:
                    val = value_function[state]
                    cell = f"{val:3.0f}"
                else:
                    cell = " · "
                
                line += cell + "|"
            
            lines.append(line)
            lines.append("─" * (cols * 4 + 1))
        
        return "\n".join(lines)
