from dataclasses import dataclass
import numpy as np


@dataclass
class GAState:
    population: np.ndarray  # shape = (N, D)
    fitness: np.ndarray  # shape = (N,)
    generation: int


def init_real_population(
        population_size: int,
        dimension: int,
        lower_bound: float,
        upper_bound: float
) -> np.ndarray:
    return np.random.uniform(
        lower_bound,
        upper_bound,
        size=(population_size, dimension)
    )


@dataclass
class SpeciesState:
    indices: np.ndarray  # indexes of individuals in population
    size: int
    role: str  # "predator" / "prey"
    mean_individual: np.ndarray  # (D,)
    mean_fitness: float


def build_species(
        population: np.ndarray,
        fitness: np.ndarray,
        clusters: list[np.ndarray]
) -> list[SpeciesState]:
    species = []
    for idx in clusters:
        if len(idx) == 0:
            continue
        species.append(
            SpeciesState(
                indices=idx,
                size=len(idx),
                role='',
                mean_individual=np.mean(population[idx], axis=0),
                mean_fitness=fitness[idx].mean()
            )
        )
    return species


if __name__ == "__main__":
   ...
