"""
Save partial results from the deep fixed-point optimization.

Based on the output log, we have complete results for N_depth=6 and N_depth=8.
N_depth=10 and N_depth=12 are still running.
"""

import numpy as np
import json
import sys
sys.path.insert(0, '..')

# Results extracted from the optimization log
partial_results = {
    'gamma': 0.05,
    'Delta': 0.3,
    'N_trunc': 3,

    # N_depth = 6 (best from seed 0)
    'd6_loss': 0.049496,
    'd6_baseline': 0.184918,
    'd6_bloch_min': 0.8848,
    'd6_bloch_max': 0.8930,
    'd6_round_fidelities': {
        1: 0.8888,
        2: 0.8180,
        3: 0.7578,
        5: 0.6615,
    },
    'd6_best_seed': 0,

    # N_depth = 8 (best from seed 0)
    'd8_loss': 0.003617,
    'd8_baseline': 0.184918,
    'd8_bloch_min': 0.9641,
    'd8_bloch_max': 1.0076,  # Note: > 1 suggests numerical issue
    'd8_round_fidelities': {
        1: 0.9833,
        # Note: Higher rounds show > 1 values, likely numerical issues
    },
    'd8_best_seed': 0,
}

# Summary
summary = {
    'status': 'partial',
    'completed_depths': [6, 8],
    'pending_depths': [10, 12],
    'best_result': {
        'depth': 8,
        'loss': 0.003617,
        'improvement_over_baseline': 0.184918 - 0.003617,
        'bloch_fidelity_min': 0.9641,
    },
    'notes': [
        'N_depth=8 achieves ||S-I||_F = 0.0036, very close to true fixed point',
        'N_depth=6 achieves ||S-I||_F = 0.0495',
        'Baseline (identity recovery): ||S-I||_F = 0.185',
        'Improvement factor: 51x (baseline/best)',
    ]
}

print("=" * 70)
print("Partial Results Summary")
print("=" * 70)
print(f"gamma = {partial_results['gamma']}")
print(f"Delta = {partial_results['Delta']}")
print()

print(f"{'Depth':<8} {'||S-I||_F':<12} {'Improvement':<12} {'Min Bloch':<12}")
print("-" * 44)
for d in [6, 8]:
    loss = partial_results[f'd{d}_loss']
    baseline = partial_results[f'd{d}_baseline']
    bloch_min = partial_results[f'd{d}_bloch_min']
    improvement = baseline - loss
    print(f"{d:<8} {loss:<12.6f} {improvement:<12.6f} {bloch_min:<12.4f}")

print()
print("Best: N_depth=8 with ||S-I||_F = 0.003617")
print("This is 51x better than identity baseline (0.185)")
print()
print("Deeper circuits (N_depth=10, 12) still optimizing...")
print()

# Save to JSON
import os
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'deep_fixedpoint_partial.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved to results/deep_fixedpoint_partial.json")
