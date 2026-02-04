import numpy as np
from typing import Callable
from opfunu.cec_based.cec2014 import F12014, F102014, F122014, F172014, F182014, F242014, F202014
from opfunu.cec_based.cec2022 import F42022, F62022, F92022, F102022


def evaluate_population(
        population: np.ndarray,
        objective: Callable[[np.ndarray], float]
) -> np.ndarray:
    """
    objective: принимает (D,) -> float
    """
    return np.apply_along_axis(objective, 1, population)


def get_0_1_fitness(fitness: np.ndarray) -> np.ndarray:
    min_fitness = fitness.min()
    max_fitness = fitness.max()
    return (fitness - min_fitness) / (max_fitness - min_fitness)


def shift_wrapper(func, shift=3):
    return lambda x: func(x - shift)


def rastrigins_function(x):
    """
    x_i in [-100; 100], i = 1,...,n
    min(f) = 0, argmin(f)_i = 0
    """
    y = 0.0512 * x
    return np.sum(y ** 2 - 10 * np.cos(2 * np.pi * y) + 10)


def griewangks_function(x: np.ndarray):
    """
    x_i in [-100; 100], i = 1,...,n
    min(f) = 0, argmin(f)_i = 0
    """
    y = 6 * x
    return 1 + np.sum(y * y / 4000) - np.prod(np.cos(y / np.sqrt(np.arange(1, y.shape[0] + 1))))


def schwefels_function(x):
    """
    x_i in [-100, 100], i = 1,...,n
    min(f) = 418.9829n, argmin(f)_i = 84,1937
    """
    if np.any(np.abs(x) > 100):
        return x.shape[0] * 1000
    y = 5 * x
    return np.sum(418.9829 - y * np.sin(np.sqrt(np.abs(y))))


def ackleys_function(x):
    """
    x_i in [-32; 23], i = 1,...,n
    min(f) = 0, argmin(f)_i = 0
    """
    return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x)))


def solomons_function(x):
    """
    x_i in [-100; 100], i = 1,...,n
    min(f) = 0, argmin(f)_i = 0
    """
    s = np.sqrt(np.sum(x ** 2))
    return 1 + 0.1 * s - np.cos(2 * np.pi * s)


def get_f12014_function(dimension):
    problem = F12014(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f122014_function(dimension):
    problem = F122014(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f242014_function(dimension):
    problem = F242014(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f102014_function(dimension):
    problem = F102014(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f172014_function(dimension):
    problem = F172014(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f182014_function(dimension):
    problem = F182014(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f202014_function(dimension):
    problem = F202014(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f42022_function(dimension):
    problem = F42022(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f62022_function(dimension):
    problem = F62022(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f92022_function(dimension):
    problem = F92022(ndim=dimension)
    return lambda x: problem.evaluate(x)


def get_f102022_function(dimension):
    problem = F102022(ndim=dimension)
    return lambda x: problem.evaluate(x)


if __name__ == "__main__":
    ...
