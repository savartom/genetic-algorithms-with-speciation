import numpy as np
from typing import Callable


def crossover_operator(
        population: np.ndarray,
        fitness: np.ndarray,
        proportion: float,
        crossover_fn: Callable,
        **kwargs
) -> np.ndarray:
    n = population.shape[0]
    if n < 2:
        return np.empty((0, *population.shape[1:]))
    num_offsprings = int(n * proportion)

    idx1 = np.random.randint(0, n, size=num_offsprings)
    idx2 = np.random.randint(0, n, size=num_offsprings)

    return crossover_fn(
        population[idx1],
        population[idx2],
        fitness[idx1],
        fitness[idx2],
        **kwargs
    )


def arithmetic_crossover(
        parents1: np.ndarray,
        parents2: np.ndarray,
        fitness1: np.ndarray,
        fitness2: np.ndarray,
        alpha: float
) -> np.ndarray:
    return alpha * parents1 + (1 - alpha) * parents2


def blend_crossover(
        parents1: np.ndarray,
        parents2: np.ndarray,
        fitness1: np.ndarray,
        fitness2: np.ndarray,
        alpha: float = 0.5
) -> np.ndarray:
    d = np.abs(parents1 - parents2)
    low = np.minimum(parents1, parents2) - alpha * d
    high = np.maximum(parents1, parents2) + alpha * d
    return np.random.uniform(low, high)


def adaptive_blend_crossover_diversity(
        parents1: np.ndarray,
        parents2: np.ndarray,
        fitness1: np.ndarray,
        fitness2: np.ndarray,
        alpha_min: float = 0.5,
        alpha_max: float = 2.5
) -> np.ndarray:
    d = np.abs(parents1 - parents2)
    diversity = np.mean(d, axis=1, keepdims=True)
    alpha = alpha_max / (1.0 + diversity)
    alpha = np.clip(alpha, alpha_min, alpha_max)

    low = np.minimum(parents1, parents2) - alpha * d
    high = np.maximum(parents1, parents2) + alpha * d
    return np.random.uniform(low, high)


def adaptive_blend_crossover_fitness(
        parents1: np.ndarray,
        parents2: np.ndarray,
        fitness1: np.ndarray,
        fitness2: np.ndarray,
        alpha_min: float = 0.5,
        alpha_max: float = 2.5
) -> np.ndarray:
    if fitness1.size == 0 or fitness2.size == 0:
        return np.empty((0, parents1.shape[1]))

    f = np.maximum(fitness1, fitness2)
    f_min, f_max = f.min(), f.max()
    norm = (f - f_min) / (f_max - f_min + 1e-8)
    alpha = alpha_min + (1.0 - norm)[:, None] * (alpha_max - alpha_min)

    d = np.abs(parents1 - parents2)
    low = np.minimum(parents1, parents2) - alpha * d
    high = np.maximum(parents1, parents2) + alpha * d
    return np.random.uniform(low, high)


if __name__ == "__main__":
    ...
