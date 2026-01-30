"""
Grid World MDP - Streamlit Application

Interactive web application for demonstrating Markov Decision Processes
using Value Iteration on a Grid World environment.

Author: AI Research Project
Date: January 2026
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, Set

# Import MDP components
from mdp import GridWorld, TransitionModel, RewardFunction
from algorithms import ValueIteration, PolicyExtractor
from simulation import Agent, Simulator
from evaluation import Metrics, Experiments
from visualization import GridRenderer, Plotter
from utils.config import *
from utils.helpers import *

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_obstacles(obstacle_str: str, grid_size: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """Parse obstacle string into set of tuples."""
    obstacles = set()
    if not obstacle_str.strip():
        return obstacles
    
    try:
        pairs = obstacle_str.split(';')
        for pair in pairs:
            pair = pair.strip()
            if pair:
                row, col = map(int, pair.split(','))
                if is_valid_state((row, col), grid_size):
                    obstacles.add((row, col))
    except:
        st.error("Invalid obstacle format. Use: row,col;row,col (e.g., 1,1;2,2)")
    
    return obstacles


@st.cache_data
def run_value_iteration(
    grid_size: Tuple[int, int],
    start_state: Tuple[int, int],
    goal_state: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    gamma: float,
    theta: float,
    stochastic: bool,
    success_prob: float,
    goal_reward: float,
    step_penalty: float
) -> Dict:
    """Run Value Iteration (cached for performance)."""
    # Create environment
    env = GridWorld(grid_size, start_state, goal_state, obstacles)
    
    # Create models
    transition_model = TransitionModel(env, stochastic, success_prob)
    reward_function = RewardFunction(env, goal_reward, step_penalty)
    
    # Run Value Iteration
    vi = ValueIteration(env, transition_model, reward_function, gamma, theta)
    result = vi.solve(verbose=False)
    
    # Extract policy
    extractor = PolicyExtractor(env, transition_model, reward_function, gamma)
    policy = extractor.extract_policy(result['value_function'])
    
    return {
        'env': env,
        'transition_model': transition_model,
        'reward_function': reward_function,
        'value_function': result['value_function'],
        'policy': policy,
        'iterations': result['iterations'],
        'converged': result['converged'],
        'convergence_history': result['convergence_history']
    }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Title and description
    st.title("🤖 Grid World MDP - Value Iteration")
    st.markdown("""
    Interactive demonstration of **Markov Decision Processes** using **Value Iteration** 
    for optimal policy computation in a Grid World navigation environment.
    """)
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Scenario selection
    st.sidebar.subheader("1. Environment Setup")
    scenario = st.sidebar.selectbox(
        "Predefined Scenario",
        options=list(SCENARIOS.keys()),
        format_func=lambda x: f"{x.title()} - {SCENARIOS[x]['description']}"
    )
    
    scenario_config = SCENARIOS[scenario]
    
    # Grid size
    col1, col2 = st.sidebar.columns(2)
    rows = col1.number_input(
        "Rows", 
        min_value=3, 
        max_value=20, 
        value=scenario_config['grid_size'][0]
    )
    cols = col2.number_input(
        "Cols", 
        min_value=3, 
        max_value=20, 
        value=scenario_config['grid_size'][1]
    )
    grid_size = (rows, cols)
    
    # Start and goal positions
    col1, col2 = st.sidebar.columns(2)
    start_row = col1.number_input("Start Row", 0, rows-1, scenario_config['start_state'][0])
    start_col = col2.number_input("Start Col", 0, cols-1, scenario_config['start_state'][1])
    start_state = (start_row, start_col)
    
    col1, col2 = st.sidebar.columns(2)
    goal_row = col1.number_input("Goal Row", 0, rows-1, scenario_config['goal_state'][0])
    goal_col = col2.number_input("Goal Col", 0, cols-1, scenario_config['goal_state'][1])
    goal_state = (goal_row, goal_col)
    
    # Obstacles
    default_obstacles = ";".join([f"{r},{c}" for r, c in scenario_config['obstacles']])
    obstacle_str = st.sidebar.text_input(
        "Obstacles (row,col;row,col)",
        value=default_obstacles,
        help="Enter obstacles as: row,col;row,col (e.g., 1,1;2,2)"
    )
    obstacles = parse_obstacles(obstacle_str, grid_size)
    
    # Algorithm parameters
    st.sidebar.subheader("2. Algorithm Parameters")
    gamma = st.sidebar.slider(
        "Discount Factor (γ)",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_GAMMA,
        step=0.05,
        help="Determines importance of future rewards"
    )
    
    theta = st.sidebar.select_slider(
        "Convergence Threshold (θ)",
        options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        value=DEFAULT_THETA,
        format_func=lambda x: f"{x:.0e}"
    )
    
    # Transition model
    st.sidebar.subheader("3. Transition Model")
    stochastic = st.sidebar.checkbox("Stochastic Transitions", value=False)
    
    if stochastic:
        success_prob = st.sidebar.slider(
            "Success Probability",
            min_value=0.5,
            max_value=1.0,
            value=DEFAULT_SUCCESS_PROB,
            step=0.05,
            help="Probability of executing intended action"
        )
    else:
        success_prob = 1.0
    
    # Reward function
    st.sidebar.subheader("4. Reward Function")
    goal_reward = st.sidebar.number_input(
        "Goal Reward",
        min_value=1.0,
        max_value=500.0,
        value=DEFAULT_GOAL_REWARD,
        step=10.0
    )
    
    step_penalty = st.sidebar.number_input(
        "Step Penalty",
        min_value=-10.0,
        max_value=-0.1,
        value=DEFAULT_STEP_PENALTY,
        step=0.1
    )
    
    # Solve button
    st.sidebar.markdown("---")
    solve_button = st.sidebar.button("🚀 Solve MDP", type="primary", use_container_width=True)
    
    # Main content area
    if solve_button or 'result' not in st.session_state:
        with st.spinner("Running Value Iteration..."):
            start_time = time.time()
            
            # Run Value Iteration
            result = run_value_iteration(
                grid_size, start_state, goal_state, obstacles,
                gamma, theta, stochastic, success_prob,
                goal_reward, step_penalty
            )
            
            elapsed_time = time.time() - start_time
            result['computation_time'] = elapsed_time
            
            # Store in session state
            st.session_state.result = result
            st.session_state.solved = True
    
    if 'solved' in st.session_state and st.session_state.solved:
        result = st.session_state.result
        
        # Display results
        st.success(f"✅ Value Iteration converged in {result['iterations']} iterations "
                  f"({format_time(result['computation_time'])})")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Iterations", result['iterations'])
        col2.metric("Discount Factor", f"{gamma:.2f}")
        col3.metric("States", len(result['env'].states))
        col4.metric("Converged", "Yes" if result['converged'] else "No")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visualizations",
            "🎯 Simulation",
            "📈 Analysis",
            "🔬 Experiments"
        ])
        
        # Tab 1: Visualizations
        with tab1:
            st.header("MDP Solution Visualization")
            
            renderer = GridRenderer(result['env'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Value Function V(s)")
                fig, ax = plt.subplots(figsize=(8, 8))
                renderer.render_value_function(result['value_function'], ax, show_values=True)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Optimal Policy π(s)")
                fig, ax = plt.subplots(figsize=(8, 8))
                renderer.render_policy(result['policy'], ax)
                st.pyplot(fig)
                plt.close()
            
            # Convergence plot
            st.subheader("Convergence History")
            plotter = Plotter()
            fig, ax = plt.subplots(figsize=(10, 6))
            plotter.plot_convergence(result['convergence_history'], ax=ax)
            st.pyplot(fig)
            plt.close()
        
        # Tab 2: Simulation
        with tab2:
            st.header("Agent Simulation")
            
            st.markdown("Execute the optimal policy and observe agent behavior.")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                num_episodes = st.number_input("Episodes", 1, 10, 1)
                simulate_button = st.button("▶️ Run Simulation", use_container_width=True)
            
            if simulate_button:
                simulator = Simulator(
                    result['env'],
                    result['transition_model'],
                    result['reward_function']
                )
                
                # Run simulation
                agent = Agent(result['env'], result['policy'])
                episode = simulator.run_episode(agent, verbose=False)
                
                # Display results
                st.success(f"Episode completed: {'Success' if episode['success'] else 'Failed'}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Steps", episode['steps'])
                col2.metric("Total Reward", f"{episode['total_reward']:.2f}")
                col3.metric("Path Length", len(episode['path']))
                
                # Visualize path
                st.subheader("Agent Path")
                renderer = GridRenderer(result['env'])
                fig, ax = plt.subplots(figsize=(10, 10))
                renderer.render_path(episode['path'], ax)
                st.pyplot(fig)
                plt.close()
                
                # Path details
                with st.expander("📋 Path Details"):
                    path_str = " → ".join([format_state(s) for s in episode['path'][:10]])
                    if len(episode['path']) > 10:
                        path_str += " → ..."
                    st.code(path_str)
                
                # Reward plot
                if episode['rewards']:
                    st.subheader("Rewards per Step")
                    plotter = Plotter()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plotter.plot_episode_rewards(episode['rewards'], ax=ax)
                    st.pyplot(fig)
                    plt.close()
        
        # Tab 3: Analysis
        with tab3:
            st.header("Performance Analysis")
            
            # Compute metrics
            simulator = Simulator(
                result['env'],
                result['transition_model'],
                result['reward_function']
            )
            
            agent = Agent(result['env'], result['policy'])
            episode = simulator.run_episode(agent, verbose=False)
            
            metrics = Metrics.compute_episode_metrics(episode)
            value_metrics = Metrics.compute_value_function_metrics(
                result['value_function'], result['env']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Episode Metrics")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                    else:
                        st.metric(key.replace('_', ' ').title(), value)
            
            with col2:
                st.subheader("Value Function Metrics")
                for key, value in value_metrics.items():
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
            
            # Policy statistics
            st.subheader("Policy Statistics")
            policy_stats = Metrics.compute_policy_metrics(result['policy'], result['env'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total States", policy_stats['total_states'])
            col2.metric("Terminal States", policy_stats['terminal_states'])
            col3.metric("Non-Terminal States", policy_stats['non_terminal_states'])
            
            # Action distribution
            if 'action_distribution' in policy_stats:
                st.subheader("Action Distribution")
                action_names = {
                    result['env'].UP: "↑ UP",
                    result['env'].DOWN: "↓ DOWN",
                    result['env'].LEFT: "← LEFT",
                    result['env'].RIGHT: "→ RIGHT"
                }
                
                dist_data = {
                    action_names.get(action, f"Action {action}"): count
                    for action, count in policy_stats['action_distribution'].items()
                }
                
                st.bar_chart(dist_data)
        
        # Tab 4: Experiments
        with tab4:
            st.header("Parameter Experiments")
            
            experiment_type = st.selectbox(
                "Experiment Type",
                ["Gamma Sweep", "Stochastic Comparison", "Grid Size Scaling"]
            )
            
            run_experiment = st.button("🔬 Run Experiment", use_container_width=True)
            
            if run_experiment:
                experiments = Experiments(result['env'])
                
                with st.spinner(f"Running {experiment_type}..."):
                    if experiment_type == "Gamma Sweep":
                        gamma_values = [0.5, 0.7, 0.9, 0.95, 0.99]
                        exp_results = experiments.gamma_sweep(
                            result['env'], gamma_values, num_trials=1, verbose=False
                        )
                        
                        st.subheader("Gamma Sweep Results")
                        
                        # Plot results
                        plotter = Plotter()
                        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                        plotter.plot_gamma_sweep(exp_results, 'avg_reward', ax=axes[0])
                        plotter.plot_gamma_sweep(exp_results, 'iterations', ax=axes[1])
                        st.pyplot(fig)
                        plt.close()
                        
                        # Results table
                        st.subheader("Detailed Results")
                        results_data = []
                        for gamma_val, data in sorted(exp_results.items()):
                            results_data.append({
                                'Gamma': gamma_val,
                                'Iterations': data['iterations'],
                                'Avg Reward': f"{data['avg_reward']:.2f}",
                                'Avg Steps': f"{data['avg_steps']:.1f}",
                                'Success Rate': f"{data['success_rate']:.1%}"
                            })
                        st.table(results_data)
                    
                    elif experiment_type == "Stochastic Comparison":
                        success_probs = [0.6, 0.7, 0.8, 0.9]
                        exp_results = experiments.stochastic_comparison(
                            result['env'], success_probs, num_episodes=5, verbose=False
                        )
                        
                        st.subheader("Stochastic vs Deterministic")
                        
                        plotter = Plotter()
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plotter.plot_stochastic_comparison(exp_results, ax=ax)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Results table
                        st.subheader("Detailed Results")
                        results_data = []
                        for config_name, data in sorted(exp_results.items()):
                            results_data.append({
                                'Configuration': config_name,
                                'Success Prob': f"{data['success_prob']:.1f}",
                                'Iterations': data['iterations'],
                                'Avg Reward': f"{data['avg_reward']:.2f}",
                                'Success Rate': f"{data['success_rate']:.1%}"
                            })
                        st.table(results_data)
                    
                    elif experiment_type == "Grid Size Scaling":
                        grid_sizes = [(5, 5), (7, 7), (10, 10)]
                        exp_results = experiments.grid_size_experiment(
                            grid_sizes, verbose=False
                        )
                        
                        st.subheader("Grid Size Scaling")
                        
                        plotter = Plotter()
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plotter.plot_grid_size_scaling(exp_results, ax=ax)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Results table
                        st.subheader("Detailed Results")
                        results_data = []
                        for size_name, data in exp_results.items():
                            results_data.append({
                                'Grid Size': size_name,
                                'States': data['num_states'],
                                'Iterations': data['iterations'],
                                'Steps': f"{data['avg_steps']:.1f}",
                                'Efficiency': f"{data['efficiency']:.1%}"
                            })
                        st.table(results_data)
    
    else:
        # Initial state - show instructions
        st.info("👈 Configure the environment and algorithm parameters in the sidebar, "
                "then click '🚀 Solve MDP' to run Value Iteration.")
        
        st.markdown("""
        ### About This Application
        
        This application demonstrates **Markov Decision Processes (MDPs)** using the 
        **Value Iteration** algorithm to find optimal policies in a Grid World environment.
        
        **Key Concepts:**
        - **States**: Grid cells representing agent positions
        - **Actions**: UP, DOWN, LEFT, RIGHT movements
        - **Transition Model**: Deterministic or stochastic state transitions
        - **Reward Function**: Goal rewards and step penalties
        - **Value Function**: Expected cumulative reward from each state
        - **Policy**: Mapping from states to optimal actions
        
        **Value Iteration** iteratively applies the Bellman optimality equation:
        
        V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]
        
        **Features:**
        - 📊 Interactive parameter configuration
        - 🎯 Real-time policy execution and simulation
        - 📈 Convergence analysis and performance metrics
        - 🔬 Parameter sweep experiments
        """)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
