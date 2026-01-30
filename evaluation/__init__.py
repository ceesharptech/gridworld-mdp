"""
Evaluation Package

Contains metrics and experiments for analyzing MDP solutions:
- Metrics: Performance and convergence metrics
- Experiments: Parameter sweeps and comparative studies
"""

from evaluation.metrics import Metrics
from evaluation.experiments import Experiments

__all__ = ['Metrics', 'Experiments']
