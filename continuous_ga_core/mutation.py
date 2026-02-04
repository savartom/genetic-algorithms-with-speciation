import numpy as np
from typing import Callable


def mutation_operator(
        population: np.ndarray,
        fitness: np.ndarray,
        proportion: float,
        mutation_fn: Callable,
        **kwargs
) -> np.ndarray:
    n = population.shape[0]
    if n < 1:
        return np.empty((0, *population.shape[1:]))
    num_mutations = int(n * proportion)

    idx = np.random.choice(n, size=num_mutations, replace=False)
    return mutation_fn(population[idx], fitness[idx], **kwargs)


def gaussian_mutation(
        population: np.ndarray,
        fitness: np.ndarray,
        sigma: float = 0.5
) -> np.ndarray:
    return population + np.random.normal(0, sigma, size=population.shape)


def uniform_mutation(
        population: np.ndarray,
        fitness: np.ndarray,
        low: float,
        high: float
) -> np.ndarray:
    return population + np.random.uniform(low, high, size=population.shape)


def adaptive_gaussian_mutation_diversity(
        population: np.ndarray,
        fitness: np.ndarray,
        sigma_min: float = 0.5,
        sigma_max: float = 2.5,
        eps: float = 1e-8
) -> np.ndarray:
    diversity = np.mean(np.std(population, axis=0))
    sigma = sigma_max / (1.0 + diversity + eps)
    sigma = np.clip(sigma, sigma_min, sigma_max)
    return population + np.random.normal(0, sigma, size=population.shape)


def adaptive_gaussian_mutation_fitness(
        population: np.ndarray,
        fitness: np.ndarray,
        sigma_min: float = 0.5,
        sigma_max: float = 2.5
) -> np.ndarray:
    if fitness.size == 0:
        return np.empty((0, population.shape[1]))

    f_min, f_max = fitness.min(), fitness.max()
    norm = (fitness - f_min) / (f_max - f_min + 1e-8)
    sigma = sigma_min + (1.0 - norm)[:, None] * (sigma_max - sigma_min)
    noise = np.random.normal(0, 1.0, size=population.shape)
    return population + sigma * noise


if __name__ == "__main__":
    ...
