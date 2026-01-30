#!/usr/bin/env python3
"""
Grid World MDP - Verification Script

This script verifies that all components are working correctly.
Run this to ensure the installation is complete and functional.

Author: AI Research Project
Date: January 2026
"""

import sys
import time

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_imports():
    """Verify all imports work correctly."""
    print_section("CHECKING IMPORTS")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        from mdp import GridWorld, TransitionModel, RewardFunction
        print("✓ MDP components imported")
    except ImportError as e:
        print(f"✗ MDP import failed: {e}")
        return False
    
    try:
        from algorithms import ValueIteration, PolicyExtractor
        print("✓ Algorithm components imported")
    except ImportError as e:
        print(f"✗ Algorithms import failed: {e}")
        return False
    
    try:
        from simulation import Agent, Simulator
        print("✓ Simulation components imported")
    except ImportError as e:
        print(f"✗ Simulation import failed: {e}")
        return False
    
    try:
        from evaluation import Metrics, Experiments
        print("✓ Evaluation components imported")
    except ImportError as e:
        print(f"✗ Evaluation import failed: {e}")
        return False
    
    try:
        from visualization import GridRenderer, Plotter
        print("✓ Visualization components imported")
    except ImportError as e:
        print(f"✗ Visualization import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test GridWorld environment."""
    print_section("TESTING ENVIRONMENT")
    
    from mdp import GridWorld
    
    try:
        # Basic environment
        env = GridWorld(grid_size=(5, 5), start_state=(0, 0), goal_state=(4, 4))
        print(f"✓ Created environment: {env}")
        
        # Check state space
        assert len(env.states) == 25, "Wrong number of states"
        print(f"✓ State space validated: {len(env.states)} states")
        
        # Check actions
        assert len(env.actions) == 4, "Wrong number of actions"
        print(f"✓ Action space validated: {len(env.actions)} actions")
        
        # Environment with obstacles
        obstacles = {(1, 1), (2, 2), (3, 3)}
        env_obs = GridWorld(
            grid_size=(5, 5),
            start_state=(0, 0),
            goal_state=(4, 4),
            obstacles=obstacles
        )
        assert len(env_obs.states) == 22, "Wrong number of states with obstacles"
        print(f"✓ Environment with obstacles: {len(env_obs.obstacles)} obstacles")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_value_iteration():
    """Test Value Iteration algorithm."""
    print_section("TESTING VALUE ITERATION")
    
    from mdp import GridWorld, TransitionModel, RewardFunction
    from algorithms import ValueIteration
    
    try:
        # Setup
        env = GridWorld(grid_size=(5, 5), start_state=(0, 0), goal_state=(4, 4))
        transition = TransitionModel(env, stochastic=False)
        reward = RewardFunction(env, goal_reward=100.0, step_penalty=-1.0)
        
        print("✓ MDP components created")
        
        # Run Value Iteration
        start_time = time.time()
        vi = ValueIteration(env, transition, reward, gamma=0.9, theta=1e-6)
        result = vi.solve(verbose=False)
        elapsed = time.time() - start_time
        
        print(f"✓ Value Iteration completed")
        print(f"  - Iterations: {result['iterations']}")
        print(f"  - Converged: {result['converged']}")
        print(f"  - Time: {elapsed*1000:.2f} ms")
        print(f"  - Final delta: {result['convergence_history'][-1]:.2e}")
        
        # Validate results
        assert result['converged'], "Failed to converge"
        assert result['iterations'] > 0, "No iterations performed"
        assert len(result['value_function']) == len(env.states), "Value function size mismatch"
        
        print("✓ Value Iteration validated")
        return True
    except Exception as e:
        print(f"✗ Value Iteration test failed: {e}")
        return False

def test_policy_extraction():
    """Test policy extraction."""
    print_section("TESTING POLICY EXTRACTION")
    
    from mdp import GridWorld, TransitionModel, RewardFunction
    from algorithms import ValueIteration, PolicyExtractor
    
    try:
        # Setup
        env = GridWorld(grid_size=(5, 5), start_state=(0, 0), goal_state=(4, 4))
        transition = TransitionModel(env, stochastic=False)
        reward = RewardFunction(env, goal_reward=100.0, step_penalty=-1.0)
        
        # Solve
        vi = ValueIteration(env, transition, reward, gamma=0.9)
        result = vi.solve(verbose=False)
        
        # Extract policy
        extractor = PolicyExtractor(env, transition, reward, gamma=0.9)
        policy = extractor.extract_policy(result['value_function'])
        
        print(f"✓ Policy extracted: {len(policy)} states")
        
        # Validate
        assert len(policy) == len(env.states), "Policy size mismatch"
        non_terminal = sum(1 for s in env.states if not env.is_terminal(s))
        assert all(policy[s] is not None for s in env.states if not env.is_terminal(s)), \
            "Missing actions for non-terminal states"
        
        print("✓ Policy validated")
        return True
    except Exception as e:
        print(f"✗ Policy extraction test failed: {e}")
        return False

def test_simulation():
    """Test simulation and agent execution."""
    print_section("TESTING SIMULATION")
    
    from mdp import GridWorld, TransitionModel, RewardFunction
    from algorithms import ValueIteration, PolicyExtractor
    from simulation import Agent, Simulator
    
    try:
        # Setup
        env = GridWorld(grid_size=(5, 5), start_state=(0, 0), goal_state=(4, 4))
        transition = TransitionModel(env, stochastic=False)
        reward = RewardFunction(env, goal_reward=100.0, step_penalty=-1.0)
        
        # Solve
        vi = ValueIteration(env, transition, reward, gamma=0.9)
        result = vi.solve(verbose=False)
        
        extractor = PolicyExtractor(env, transition, reward, gamma=0.9)
        policy = extractor.extract_policy(result['value_function'])
        
        # Simulate
        simulator = Simulator(env, transition, reward, max_steps=100)
        agent = Agent(env, policy)
        episode = simulator.run_episode(agent, verbose=False)
        
        print(f"✓ Simulation completed")
        print(f"  - Success: {episode['success']}")
        print(f"  - Steps: {episode['steps']}")
        print(f"  - Total Reward: {episode['total_reward']:.2f}")
        print(f"  - Path Length: {len(episode['path'])}")
        
        # Validate
        assert episode['success'], "Agent failed to reach goal"
        assert episode['steps'] > 0, "No steps taken"
        assert len(episode['path']) == episode['steps'] + 1, "Path length mismatch"
        
        print("✓ Simulation validated")
        return True
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return False

def test_stochastic():
    """Test stochastic transitions."""
    print_section("TESTING STOCHASTIC TRANSITIONS")
    
    from mdp import GridWorld, TransitionModel, RewardFunction
    from algorithms import ValueIteration
    
    try:
        env = GridWorld(grid_size=(5, 5), start_state=(0, 0), goal_state=(4, 4))
        
        # Stochastic model
        transition = TransitionModel(env, stochastic=True, success_prob=0.8)
        reward = RewardFunction(env)
        
        print(f"✓ Stochastic model created: {transition}")
        
        # Test transition probabilities
        state = (2, 2)
        action = env.UP
        transitions = transition.get_possible_transitions(state, action)
        
        print(f"✓ Transitions from {state} with UP action:")
        total_prob = 0.0
        for next_state, prob in transitions:
            print(f"    → {next_state}: {prob:.2f}")
            total_prob += prob
        
        assert abs(total_prob - 1.0) < 1e-10, "Probabilities don't sum to 1"
        print(f"✓ Probabilities sum to {total_prob:.4f}")
        
        # Solve with stochastic model
        vi = ValueIteration(env, transition, reward, gamma=0.9)
        result = vi.solve(verbose=False)
        
        print(f"✓ Stochastic Value Iteration converged in {result['iterations']} iterations")
        
        return True
    except Exception as e:
        print(f"✗ Stochastic test failed: {e}")
        return False

def test_experiments():
    """Test experiments module."""
    print_section("TESTING EXPERIMENTS")
    
    from mdp import GridWorld
    from evaluation import Experiments
    
    try:
        env = GridWorld(grid_size=(5, 5), start_state=(0, 0), goal_state=(4, 4))
        experiments = Experiments(env)
        
        print("✓ Experiments module initialized")
        
        # Gamma sweep
        gamma_results = experiments.gamma_sweep(
            env,
            gamma_values=[0.7, 0.9],
            num_trials=1,
            verbose=False
        )
        
        print(f"✓ Gamma sweep completed: {len(gamma_results)} configurations")
        
        for gamma, data in gamma_results.items():
            print(f"    γ={gamma}: {data['iterations']} iterations, "
                  f"reward={data['avg_reward']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Experiments test failed: {e}")
        return False

def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("  GRID WORLD MDP - VERIFICATION SCRIPT")
    print("="*70)
    print("\nThis script verifies all components are working correctly.")
    
    tests = [
        ("Import Check", check_imports),
        ("Environment Test", test_environment),
        ("Value Iteration Test", test_value_iteration),
        ("Policy Extraction Test", test_policy_extraction),
        ("Simulation Test", test_simulation),
        ("Stochastic Transitions Test", test_stochastic),
        ("Experiments Test", test_experiments)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} encountered unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print("\n" + "-"*70)
    print(f"Results: {passed}/{total} tests passed")
    print("-"*70)
    
    if passed == total:
        print("\n🎉 All tests passed! System is fully operational.")
        print("\nYou can now run the application with:")
        print("    streamlit run app.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
