# Grid World MDP - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd gridworld_mdp_app
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run app.py
```

### Step 3: Explore!
The app will open at http://localhost:8501

## 📋 Quick Test (Command Line)

Test the complete system without the UI:

```bash
cd gridworld_mdp_app
python -c "
from mdp import GridWorld, TransitionModel, RewardFunction
from algorithms import ValueIteration, PolicyExtractor
from simulation import Agent, Simulator

# Create 5x5 grid
env = GridWorld(grid_size=(5,5), start_state=(0,0), goal_state=(4,4))

# Setup MDP
transition = TransitionModel(env, stochastic=False)
reward = RewardFunction(env, goal_reward=100.0, step_penalty=-1.0)

# Solve with Value Iteration
vi = ValueIteration(env, transition, reward, gamma=0.9)
result = vi.solve(verbose=True)

# Extract and execute policy
extractor = PolicyExtractor(env, transition, reward, gamma=0.9)
policy = extractor.extract_policy(result['value_function'])

simulator = Simulator(env, transition, reward)
agent = Agent(env, policy)
episode = simulator.run_episode(agent, verbose=True)

print(f'\n✅ Success! Agent reached goal in {episode[\"steps\"]} steps')
print(f'Total reward: {episode[\"total_reward\"]:.2f}')
"
```

## 🎯 Application Features

### 1. Main Interface
- **Sidebar**: Configure environment, algorithm parameters, rewards
- **Solve Button**: Run Value Iteration
- **4 Tabs**: Visualizations, Simulation, Analysis, Experiments

### 2. Predefined Scenarios
- **Simple**: 5×5 grid with one obstacle
- **Corridor**: Maze requiring pathfinding
- **Complex**: 10×10 with multiple obstacles
- **Open**: No obstacles, shortest path demo

### 3. Key Parameters

**Discount Factor (γ)**: 0.1 to 1.0
- Lower values: Prioritize immediate rewards
- Higher values: Consider long-term consequences

**Convergence Threshold (θ)**: 1e-2 to 1e-7
- Determines stopping precision
- Smaller = more iterations but higher accuracy

**Transition Model**:
- Deterministic: Perfect control
- Stochastic: Realistic uncertainty (0.5-1.0 success probability)

**Rewards**:
- Goal Reward: 1-500 (default: 100)
- Step Penalty: -10 to -0.1 (default: -1)

## 📊 Tabs Overview

### Visualizations Tab
- **Value Function**: Heatmap showing V(s) for all states
- **Policy Arrows**: Optimal action at each state
- **Convergence Plot**: δ over iterations

### Simulation Tab
- Execute the policy with agent animation
- View step-by-step trajectory
- Track rewards per step
- Compare to optimal path length

### Analysis Tab
- Episode metrics (steps, rewards, success rate)
- Value function statistics
- Policy distribution (action frequencies)

### Experiments Tab
- **Gamma Sweep**: Test different discount factors
- **Stochastic Comparison**: Deterministic vs stochastic
- **Grid Size Scaling**: Performance on larger grids

## 🔧 Programmatic Usage

### Basic Example
```python
from mdp import GridWorld, TransitionModel, RewardFunction
from algorithms import ValueIteration, PolicyExtractor

# Setup
env = GridWorld(grid_size=(8,8), start_state=(0,0), goal_state=(7,7))
transition = TransitionModel(env)
reward = RewardFunction(env)

# Solve
vi = ValueIteration(env, transition, reward, gamma=0.95)
result = vi.solve()

# Get policy
extractor = PolicyExtractor(env, transition, reward, gamma=0.95)
policy = extractor.extract_policy(result['value_function'])
```

### Custom Obstacles
```python
obstacles = {(2,2), (2,3), (3,2), (3,3)}
env = GridWorld(
    grid_size=(10,10),
    start_state=(0,0),
    goal_state=(9,9),
    obstacles=obstacles
)
```

### Stochastic Environment
```python
transition = TransitionModel(
    env,
    stochastic=True,
    success_prob=0.8  # 80% intended, 10% each perpendicular
)
```

### Run Experiments
```python
from evaluation import Experiments

experiments = Experiments(env)

# Gamma sweep
results = experiments.gamma_sweep(
    env,
    gamma_values=[0.5, 0.7, 0.9, 0.95, 0.99],
    num_trials=10
)
```

## 📈 Expected Results

### Typical Performance (5×5 Grid)
- Iterations to converge: 8-15
- Optimal path length: ~8 steps (Manhattan distance)
- Total reward: ~92 (100 goal - 8 steps)
- Computation time: <100ms

### Larger Grids (10×10)
- Iterations: 20-40
- Optimal path: ~18 steps
- Computation time: <500ms

## 🐛 Troubleshooting

**Import errors**: Make sure you're in the `gridworld_mdp_app` directory

**Streamlit issues**: Update streamlit: `pip install --upgrade streamlit`

**Slow performance**: Reduce grid size or increase convergence threshold

**No convergence**: Check that goal is reachable (not blocked by obstacles)

## 📚 Learn More

- **README.md**: Complete documentation
- **Source code**: Heavily commented with docstrings
- **Module docs**: Each package has detailed explanations

## 🎓 Academic Usage

This codebase is designed for:
- AI/ML course projects
- Research paper implementations
- Algorithm comparisons
- Educational demonstrations

**Citation**: When using in academic work, cite as implementation of Value Iteration algorithm for MDP optimal policy computation.

## ✅ Verification Checklist

Run these to verify installation:

```bash
# Check Python version (3.10+ required)
python --version

# Check dependencies
pip list | grep -E "numpy|matplotlib|streamlit"

# Test imports
python -c "from mdp import GridWorld; print('✓ Imports OK')"

# Run full test
python -c "
from mdp import GridWorld, TransitionModel, RewardFunction
from algorithms import ValueIteration

env = GridWorld()
trans = TransitionModel(env)
rew = RewardFunction(env)
vi = ValueIteration(env, trans, rew)
result = vi.solve(verbose=False)
assert result['converged'], 'Failed to converge'
print(f'✓ Converged in {result[\"iterations\"]} iterations')
"
```

All checks should pass! ✅

---

**Need Help?** Check the comprehensive README.md or examine the well-documented source code.

**Ready to Start?** Run `streamlit run app.py` and explore!
