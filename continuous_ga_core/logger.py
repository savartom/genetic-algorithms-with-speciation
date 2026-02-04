import numpy as np
from continuous_ga_core.population import GAState, SpeciesState


class DataLogger:
    def __init__(self):
        self.generations = []
        self.best_fitness = []
        self.mean_fitness = []
        self.data = []

    def log(self, state: GAState):
        self.generations.append(state.generation)
        self.best_fitness.append(np.min(state.fitness))
        self.mean_fitness.append(np.mean(state.fitness))

    def end(self):
        min_fitness = min(self.best_fitness)
        print(f"min fitness: {min_fitness}")

        self.data.append((self.generations, self.best_fitness))

        self.generations = []
        self.best_fitness = []
        self.mean_fitness = []

    def clear(self):
        self.data.clear()
        self.generations.clear()
        self.best_fitness.clear()
        self.mean_fitness.clear()


class SpeciesLoggerForOneSpecies:
    def __init__(self):
        self.generations = []
        self.fitness = []
        self.roles = []

    def log(
            self,
            state: GAState
    ) -> None:
        self.generations.append([state.population])
        self.fitness.append([state.fitness])
        self.roles.append([None])

    def clear(self):
        self.generations.clear()
        self.fitness.clear()
        self.roles.clear()


class SpeciesLogger:
    def __init__(self):
        self.generations = []
        self.fitness = []
        self.roles = []

    def log(
            self,
            state: GAState,
            species: list[SpeciesState]
    ) -> None:
        self.generations.append([state.population[sp.indices] for sp in species])
        self.fitness.append([state.fitness[sp.indices] for sp in species])
        self.roles.append([sp.role for sp in species])

    def clear(self):
        self.generations.clear()
        self.fitness.clear()
        self.roles.clear()


if __name__ == "__main__":
    ...
