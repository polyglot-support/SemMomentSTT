import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, Tuple, List

def evaluate_parameters(
    initial_position: np.ndarray,
    target_position: np.ndarray,
    params: Dict[str, float],
    n_steps: int = 5
) -> Tuple[float, Dict[str, float]]:
    """
    Simulate trajectory updates and evaluate parameter performance.

    Args:
        initial_position: Starting position vector.
        target_position: Target position vector.
        params: Parameters to evaluate (force_scale, step_size, momentum_decay).
        n_steps: Number of simulation steps.

    Returns:
        Tuple of (score, metrics).
    """
    position = initial_position.copy()
    momentum = np.zeros_like(position)

    position_changes = []
    momentum_magnitudes = []
    similarities = []

    for step in range(n_steps):
        if step == 0:
            force = target_position - position
            force = np.clip(force * params['force_scale'], -1.0, 1.0)
            momentum = force * params['step_size']
        else:
            momentum *= params['momentum_decay']

        position += momentum

        position_changes.append(np.linalg.norm(position - initial_position))
        momentum_magnitudes.append(np.linalg.norm(momentum))
        similarities.append(1 - cosine(position, target_position))

    metrics = {
        'position_change': position_changes[-1],
        'momentum_decay': all(m1 > m2 for m1, m2 in zip(momentum_magnitudes[:-1], momentum_magnitudes[1:])),
        'final_similarity': similarities[-1],
        'initial_momentum': 0.0,
        'final_momentum': momentum_magnitudes[-1],
        'max_momentum': max(momentum_magnitudes),
    }

    score = (
        metrics['position_change'] +
        (3.0 if metrics['momentum_decay'] else 0.0) +
        metrics['final_similarity'] +
        (1.0 if metrics['initial_momentum'] == 0.0 else 0.0) +
        (2.0 if metrics['max_momentum'] > 0.0 else 0.0)
    )

    return score, metrics

def grid_search() -> List[Tuple[Dict[str, float], float, Dict[str, float]]]:
    """
    Perform a grid search to optimize parameters.

    Returns:
        List of (params, score, metrics) tuples, sorted by score.
    """
    param_ranges = {
        'force_scale': np.linspace(0.5, 20, 10),
        'step_size': np.linspace(1, 200, 10),
        'momentum_decay': np.linspace(0.999, 1, 10),
    }

    dim = 768
    initial_position = np.random.randn(dim)
    initial_position /= np.linalg.norm(initial_position)
    target_position = -initial_position

    results = []
    total_combinations = len(param_ranges['force_scale']) * len(param_ranges['step_size']) * len(param_ranges['momentum_decay'])

    print("Starting grid search...")
    print(f"Testing {total_combinations} parameter combinations")

    for force_scale in param_ranges['force_scale']:
        for step_size in param_ranges['step_size']:
            for momentum_decay in param_ranges['momentum_decay']:
                params = {
                    'force_scale': force_scale,
                    'step_size': step_size,
                    'momentum_decay': momentum_decay
                }
                score, metrics = evaluate_parameters(initial_position, target_position, params)
                results.append((params, score, metrics))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main():
    """Run the grid search and display results."""
    results = grid_search()

    print("\nTop 5 parameter combinations:")
    print("Rank  Force Scale  Step Size  Momentum Decay  Score")
    print("-" * 50)

    for i, (params, score, _) in enumerate(results[:5]):
        print(f"{i + 1:4d}  {params['force_scale']:10.3f}  {params['step_size']:9.3f}  "
              f"{params['momentum_decay']:14.3f}  {score:6.3f}")

    best_params, best_score, best_metrics = results[0]

    print("\nBest parameters found:")
    print(f"Force scale: {best_params['force_scale']:.3f}")
    print(f"Step size: {best_params['step_size']:.3f}")
    print(f"Momentum decay: {best_params['momentum_decay']:.3f}")
    print(f"Score: {best_score:.3f}")

    print("\nMetrics:")
    for key, value in best_metrics.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")

if __name__ == '__main__':
    main()
