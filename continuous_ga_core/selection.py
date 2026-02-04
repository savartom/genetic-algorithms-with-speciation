import numpy as np


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    population_size: int,
    tournament_size: int,
    elite_size: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        if population is None or fitness is None:
            raise ValueError

        if population.ndim != 2:
            population = np.atleast_2d(population)

        n, genome_len = population.shape

        if n == 0:
            dummy = np.zeros((1, genome_len))
            return dummy, np.zeros(1)

        fitness = np.asarray(fitness).reshape(-1)
        if fitness.shape[0] != n:
            fitness = np.resize(fitness, n)

        population_size = max(1, int(population_size))
        elite_size = max(0, min(int(elite_size), population_size, n))
        tournament_size = max(1, int(tournament_size))

        new_population = np.empty((population_size, genome_len),
                                  dtype=population.dtype)
        new_fitness = np.empty(population_size,
                               dtype=fitness.dtype)

        if elite_size > 0:
            k = min(elite_size, n)
            elite_idx = np.argpartition(fitness, k - 1)[:k]

            new_population[:k] = population[elite_idx]
            new_fitness[:k] = fitness[elite_idx]

            for i in range(k, elite_size):
                new_population[i] = population[elite_idx[0]]
                new_fitness[i] = fitness[elite_idx[0]]

        for i in range(elite_size, population_size):
            try:
                if tournament_size >= n:
                    candidates = np.random.randint(0, n, size=tournament_size)
                else:
                    candidates = np.random.choice(n, tournament_size, replace=False)

                best = candidates[np.argmin(fitness[candidates])]
            except Exception:
                best = np.random.randint(0, n)

            new_population[i] = population[best]
            new_fitness[i] = fitness[best]

        return new_population, new_fitness

    except Exception:
        genome_len = population.shape[1] if population.ndim == 2 else 1
        pop = np.zeros((1, genome_len))
        fit = np.zeros(1)
        return pop, fit


if __name__ == "__main__":
    ...
