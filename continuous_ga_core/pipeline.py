import math

import numpy as np
from typing import Callable, Optional, Any

from continuous_ga_core.interaction import evolve_species, recalc_params
from continuous_ga_core.logger import DataLogger, SpeciesLogger, SpeciesLoggerForOneSpecies
from continuous_ga_core.population import GAState, init_real_population, build_species
from continuous_ga_core.evaluation import evaluate_population, get_0_1_fitness
from utils.timeit import timeit


class GeneticAlgorithm:
    def __init__(
            self,
            init_fn: Callable,
            selection_fn: Callable,
            crossover_fn: Callable,
            mutation_fn: Callable,
            termination_fn: Callable
    ):
        self.population_size = None
        self.dimension = None
        self.limit = None
        self.objective = None
        self.init_fn = init_fn
        self.selection_fn = selection_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.termination_fn = termination_fn
        self.logger = None
        self.species_logger = None

    @timeit
    def run(self,
            population_size: int,
            dimension: int,
            limit: int,
            func: Callable[[np.ndarray], float],
            logger: Optional[DataLogger] = None,
            species_logger: Optional[SpeciesLoggerForOneSpecies] = None
            ) -> np.ndarray:
        self.population_size = population_size
        self.dimension = dimension
        self.limit = limit
        self.objective = func
        self.logger = logger
        self.species_logger = species_logger

        population = self.init_fn(self.population_size, self.dimension)
        fitness = evaluate_population(population, self.objective)
        state = GAState(population, fitness, generation=0)

        if self.logger:
            self.logger.log(state)

        while not self.termination_fn(state, self.limit):
            if self.species_logger:
                self.species_logger.log(state)

            offsprings_c = self.crossover_fn(population, fitness)
            offsprings_m = self.mutation_fn(population, fitness)

            population = np.vstack([population, offsprings_c, offsprings_m])
            fitness = evaluate_population(population, self.objective)

            population, fitness = self.selection_fn(population, fitness, self.population_size)

            state = GAState(population, fitness, state.generation + 1)

            if self.logger:
                self.logger.log(state)

        if self.logger:
            self.logger.end()

        best_idx = np.argmin(state.fitness)
        return state.fitness[best_idx], state.population[best_idx]


class GeneticAlgorithmWithSpeciation:
    def __init__(
            self,
            init_fn: Callable,
            selection_fn: Callable,
            crossover_fn: Callable,
            mutation_fn: Callable,
            clusterization_fn: Callable,
            role_assignment_fn: Callable,
            interaction_fn: Callable,
            termination_fn: Callable
    ):
        self.population_size = None
        self.dimension = None
        self.limit = None
        self.objective = None
        self.init_fn = init_fn
        self.selection_fn = selection_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.clusterization_fn = clusterization_fn
        self.interaction_fn = interaction_fn
        self.role_assignment_fn = role_assignment_fn
        self.termination_fn = termination_fn
        self.logger = None
        self.species_logger = None

    @timeit
    def run(self,
            population_size: int,
            dimension: int,
            limit: int,
            func: Callable[[np.ndarray], float],
            logger: Optional[DataLogger] = None,
            species_logger: Optional[SpeciesLogger] = None
            ) -> np.ndarray:
        self.population_size = population_size
        self.dimension = dimension
        self.limit = limit
        self.objective = func
        self.logger = logger
        self.species_logger = species_logger

        population = self.init_fn(self.population_size, self.dimension)
        fitness = evaluate_population(population, self.objective)
        fitness_0_1 = get_0_1_fitness(fitness)
        state = GAState(population, fitness, generation=0)

        if self.logger:
            self.logger.log(state)

        while not self.termination_fn(state, self.limit):
            clusters = self.clusterization_fn(population)
            species = build_species(population, fitness_0_1, clusters)

            self.role_assignment_fn(species)

            if self.species_logger:
                self.species_logger.log(state, species)

            for sp in species:
                population = evolve_species(
                    population=population,
                    fitness=fitness,
                    species=sp,
                    crossover_fn=self.crossover_fn,
                    mutation_fn=self.mutation_fn
                )

            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)

            for sp in species:
                recalc_params(
                    population=population,
                    fitness=fitness_0_1,
                    species=sp
                )

            fitness_0_1 = self.interaction_fn(population, fitness_0_1, species)

            new_population = []
            for sp in species:
                new_pop, new_fit = self.selection_fn(
                    population[sp.indices],
                    fitness_0_1[sp.indices],
                    sp.size
                )
                new_population.append(new_pop)

            population = np.concatenate(new_population)
            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)
            state = GAState(population, fitness, state.generation + 1)

            if self.logger:
                self.logger.log(state)

        if self.logger:
            self.logger.end()

        best_idx = np.argmin(state.fitness)
        return state.fitness[best_idx], state.population[best_idx]


class GeneticAlgorithmWithSpeciationFast:
    def __init__(
            self,
            init_fn: Callable,
            selection_fn: Callable,
            crossover_fn: Callable,
            mutation_fn: Callable,
            clusterization_fn: Callable,
            role_assignment_fn: Callable,
            interaction_fn: Callable,
            termination_fn: Callable
    ):
        self.population_size = None
        self.dimension = None
        self.limit = None
        self.objective = None
        self.init_fn = init_fn
        self.selection_fn = selection_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.clusterization_fn = clusterization_fn
        self.interaction_fn = interaction_fn
        self.role_assignment_fn = role_assignment_fn
        self.termination_fn = termination_fn

    def run(self,
            population_size: int,
            dimension: int,
            limit: int,
            func: Callable[[np.ndarray], float]
            ) -> tuple[np.ndarray[tuple[Any, ...], Any], np.ndarray[tuple[Any, ...], Any]]:
        self.population_size = population_size
        self.dimension = dimension
        self.limit = limit
        self.objective = func

        population = self.init_fn(self.population_size, self.dimension)
        fitness = evaluate_population(population, self.objective)
        fitness_0_1 = get_0_1_fitness(fitness)
        state = GAState(population, fitness, generation=0)

        while not self.termination_fn(state, self.limit):
            clusters = self.clusterization_fn(population)
            species = build_species(population, fitness_0_1, clusters)

            self.role_assignment_fn(species)

            for sp in species:
                population = evolve_species(
                    population=population,
                    fitness=fitness,
                    species=sp,
                    crossover_fn=self.crossover_fn,
                    mutation_fn=self.mutation_fn
                )

            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)

            for sp in species:
                recalc_params(
                    population=population,
                    fitness=fitness_0_1,
                    species=sp
                )

            fitness_0_1 = self.interaction_fn(population, fitness_0_1, species)

            new_population = []
            for sp in species:
                new_pop, new_fit = self.selection_fn(
                    population[sp.indices],
                    fitness_0_1[sp.indices],
                    sp.size
                )
                new_population.append(new_pop)

            population = np.concatenate(new_population)
            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)
            state = GAState(population, fitness, state.generation + 1)

        best_idx = np.argmin(state.fitness)
        return state.fitness[best_idx], state.population[best_idx]


class GeneticAlgorithmWithSpeciationAdaptive:
    def __init__(
            self,
            init_fn: Callable,
            selection_fn: Callable,
            crossover_fn: Callable,
            mutation_fn: Callable,
            clusterization_fn: Callable,
            role_assignment_fn: Callable,
            interaction_fn: Callable,
            termination_fn: Callable
    ):
        self.population_size = None
        self.dimension = None
        self.limit = None
        self.objective = None
        self.init_fn = init_fn
        self.selection_fn = selection_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.clusterization_fn = clusterization_fn
        self.interaction_fn = interaction_fn
        self.role_assignment_fn = role_assignment_fn
        self.termination_fn = termination_fn
        self.logger = None
        self.species_logger = None

    @timeit
    def run(self,
            population_size: int,
            dimension: int,
            limit: int,
            func: Callable[[np.ndarray], float],
            logger: Optional[DataLogger] = None,
            species_logger: Optional[SpeciesLogger] = None
            ) -> np.ndarray:
        self.population_size = population_size
        self.dimension = dimension
        self.limit = limit
        self.objective = func
        self.logger = logger
        self.species_logger = species_logger

        population = self.init_fn(self.population_size, self.dimension)
        fitness = evaluate_population(population, self.objective)
        fitness_0_1 = get_0_1_fitness(fitness)
        state = GAState(population, fitness, generation=0)

        if self.logger:
            self.logger.log(state)

        while not self.termination_fn(state, self.limit):
            clusters = self.clusterization_fn(population)
            species = build_species(population, fitness_0_1, clusters)

            self.role_assignment_fn(species)

            if self.species_logger:
                self.species_logger.log(state, species)

            for sp in species:
                population = evolve_species(
                    population=population,
                    fitness=fitness,
                    species=sp,
                    crossover_fn=self.crossover_fn,
                    mutation_fn=self.mutation_fn
                )

            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)

            for sp in species:
                recalc_params(
                    population=population,
                    fitness=fitness_0_1,
                    species=sp
                )

            fitness_0_1 = self.interaction_fn(population, fitness_0_1, species,
                                              state.generation / self.limit)

            new_population = []
            for sp in species:
                new_pop, new_fit = self.selection_fn(
                    population[sp.indices],
                    fitness_0_1[sp.indices],
                    sp.size
                )
                new_population.append(new_pop)

            population = np.concatenate(new_population)
            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)
            state = GAState(population, fitness, state.generation + 1)

            if self.logger:
                self.logger.log(state)

        if self.logger:
            self.logger.end()

        best_idx = np.argmin(state.fitness)
        return state.fitness[best_idx], state.population[best_idx]


class GeneticAlgorithmWithSpeciation2:
    def __init__(
            self,
            init_fn: Callable,
            selection_fn: Callable,
            crossover_fn: Callable,
            mutation_fn: Callable,
            clusterization_fn: Callable,
            role_assignment_fn: Callable,
            interaction_fn: Callable,
            alpha: float,
            gamma: float,
            termination_fn: Callable
    ):
        self.population_size = None
        self.dimension = None
        self.limit = None
        self.objective = None
        self.init_fn = init_fn
        self.selection_fn = selection_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.clusterization_fn = clusterization_fn
        self.interaction_fn = interaction_fn
        self.alpha_start = alpha
        self.gamma_start = gamma
        self.role_assignment_fn = role_assignment_fn
        self.termination_fn = termination_fn
        self.logger = None
        self.species_logger = None

    @timeit
    def run(self,
            population_size: int,
            dimension: int,
            limit: int,
            func: Callable[[np.ndarray], float],
            logger: Optional[DataLogger] = None,
            species_logger: Optional[SpeciesLogger] = None
            ) -> np.ndarray:
        self.population_size = population_size
        self.dimension = dimension
        self.limit = limit
        self.objective = func
        self.logger = logger
        self.species_logger = species_logger

        alpha = self.alpha_start
        gamma = self.gamma_start

        population = self.init_fn(self.population_size, self.dimension)
        fitness = evaluate_population(population, self.objective)
        fitness_0_1 = get_0_1_fitness(fitness)
        state = GAState(population, fitness, generation=0)

        if self.logger:
            self.logger.log(state)

        while not self.termination_fn(state, self.limit):
            clusters = self.clusterization_fn(population)
            species = build_species(population, fitness_0_1, clusters)

            self.role_assignment_fn(species)

            if self.species_logger:
                self.species_logger.log(state, species)

            for sp in species:
                population = evolve_species(
                    population=population,
                    fitness=fitness,
                    species=sp,
                    crossover_fn=self.crossover_fn,
                    mutation_fn=self.mutation_fn
                )

            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)

            for sp in species:
                recalc_params(
                    population=population,
                    fitness=fitness_0_1,
                    species=sp
                )

            fitness_0_1, alpha, gamma = self.interaction_fn(population, fitness_0_1, species, alpha, gamma)

            new_population = []
            for sp in species:
                new_pop, new_fit = self.selection_fn(
                    population[sp.indices],
                    fitness_0_1[sp.indices],
                    sp.size
                )
                new_population.append(new_pop)

            population = np.concatenate(new_population)
            fitness = evaluate_population(population, self.objective)
            fitness_0_1 = get_0_1_fitness(fitness)
            state = GAState(population, fitness, state.generation + 1)

            if self.logger:
                self.logger.log(state)

        if self.logger:
            self.logger.end()

        best_idx = np.argmin(state.fitness)
        return state.fitness[best_idx], state.population[best_idx]


if __name__ == "__main__":
    from evaluation import griewangks_function, shift_wrapper, get_f12014_function, \
        schwefels_function
    from selection import tournament_selection
    from crossover import crossover_operator, blend_crossover, adaptive_blend_crossover_diversity
    from mutation import mutation_operator, gaussian_mutation, adaptive_gaussian_mutation_diversity
    from termination import termination_by_generation
    from clusterization import (hdbscan_clusterization_indices, kmeans_clusterization_indices,
                                dbscan_clusterization_indices, optics_clusterization_indices)
    from interaction import (assign_roles_random, assign_roles_power, assign_roles_median,
                             predator_prey_interaction,
                             predator_prey_interaction_with_mean_fitness_update,
                             predator_prey_interaction_with_mean_fitness_update_with_solve_and_scale,
                             predator_prey_interaction_with_mean_fitness_update_and_scale)
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    import random


    settings = {
        'griewangks 2D 100 individuals': {
            'title': 'griewangks 2D 100 individuals',
            'population_size': 100,
            'dimension': 2,
            'limit': 500,
            'function': griewangks_function
        },
        'shifted griewangks 2D 500 individuals': {
            'title': 'shifted griewangks 2D 500 individuals',
            'population_size': 500,
            'dimension': 2,
            'limit': 500,
            'function': shift_wrapper(griewangks_function, shift=25)
        },
        'griewangks 10D 1000 individuals': {
            'title': 'griewangks 10D 1000 individuals',
            'population_size': 100,
            'dimension': 2,
            'limit': 500,
            'function': griewangks_function
        },

    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    random.seed(241)
    np.random.seed(241)

    logger = DataLogger()
    species_logger_for_classic_ga = SpeciesLoggerForOneSpecies()
    species_logger = SpeciesLogger()

    population_size = 50
    dimension = 2
    limit = 300

    func = schwefels_function

    # func = shift_wrapper(griewangks_function, shift=25)
    # func = get_f12014_function(dimension) #shift_wrapper(griewangks_function, shift=25)

    ga = GeneticAlgorithm(
        init_fn=lambda pop_size, dim: init_real_population(
            pop_size,
            dim,
            lower_bound=-100,
            upper_bound=100
        ),
        selection_fn=lambda pop, fit, size: tournament_selection(
            pop,
            fit,
            population_size=size,
            tournament_size=2,
            elite_size=math.ceil(size * 0.02)
        ),
        crossover_fn=lambda pop, fit: crossover_operator(
            pop,
            fit,
            proportion=0.5,
            crossover_fn=blend_crossover,
            alpha=0.5
        ),
        mutation_fn=lambda pop, fit: mutation_operator(
            pop,
            fit,
            proportion=0.5,
            mutation_fn=gaussian_mutation,
            sigma=0.1
        ),
        termination_fn=lambda state, lim: termination_by_generation(
            state,
            lim
        )
    )

    gas0 = GeneticAlgorithmWithSpeciation(
        init_fn=lambda pop_size, dim: init_real_population(
            pop_size,
            dim,
            lower_bound=-100,
            upper_bound=100
        ),
        selection_fn=lambda pop, fit, size: tournament_selection(
            pop,
            fit,
            population_size=size,
            tournament_size=2,
            elite_size=math.ceil(size * 0.02)
        ),
        crossover_fn=lambda pop, fit: crossover_operator(
            pop,
            fit,
            proportion=0.6,
            crossover_fn=adaptive_blend_crossover_diversity,
            alpha_min=0.4,
            alpha_max=2.5
        ),
        mutation_fn=lambda pop, fit: mutation_operator(
            pop,
            fit,
            proportion=0.6,
            mutation_fn=adaptive_gaussian_mutation_diversity,
            sigma_min=0.6,
            sigma_max=2.5
        ),
        clusterization_fn=lambda pop:
        # dbscan_clusterization_indices(
        #    population=pop,
        #    eps=0.4,
        #    min_samples=5
        # ),
        hdbscan_clusterization_indices(
            population=pop,
            min_cluster_size=10
        ),
        # kmeans_clusterization_indices(
        #    population=pop,
        #    n_clusters=8
        # ),
        # optics_clusterization_indices(
        #    population=pop,
        #    xi=0.3,
        #    min_samples=5,
        #    min_cluster_size=10
        # ),
        role_assignment_fn=lambda sp: assign_roles_median(
            species=sp
        ),
        interaction_fn=lambda pop, fit, sp: predator_prey_interaction(
            population=pop,
            fitness=fit,
            species=sp,
            alpha=0.4 * 1,
            gamma=-0.8 * 1,
            sigma=120
        ),
        termination_fn=lambda state, lim: termination_by_generation(
            state,
            lim
        )
    )

    gas1 = GeneticAlgorithmWithSpeciation(
        init_fn=lambda pop_size, dim: init_real_population(
            pop_size,
            dim,
            lower_bound=-100,
            upper_bound=100
        ),
        selection_fn=lambda pop, fit, size: tournament_selection(
            pop,
            fit,
            population_size=size,
            tournament_size=2,
            elite_size=math.ceil(size * 0.02)
        ),
        crossover_fn=lambda pop, fit: crossover_operator(
            pop,
            fit,
            proportion=0.6,
            crossover_fn=adaptive_blend_crossover_diversity,
            alpha_min=0.4,
            alpha_max=2.5
        ),
        mutation_fn=lambda pop, fit: mutation_operator(
            pop,
            fit,
            proportion=0.6,
            mutation_fn=adaptive_gaussian_mutation_diversity,
            sigma_min=0.6,
            sigma_max=2.5
        ),
        clusterization_fn=lambda pop:
        # dbscan_clusterization_indices(
        #    population=pop,
        #    eps=0.4,
        #    min_samples=5
        # ),
        hdbscan_clusterization_indices(
            population=pop,
            min_cluster_size=10
        ),
        # kmeans_clusterization_indices(
        #    population=pop,
        #    n_clusters=8
        # ),
        # optics_clusterization_indices(
        #    population=pop,
        #    xi=0.3,
        #    min_samples=5,
        #    min_cluster_size=10
        # ),
        role_assignment_fn=lambda sp: assign_roles_median(
            species=sp
        ),
        interaction_fn=lambda pop, fit,
                              sp: predator_prey_interaction_with_mean_fitness_update(
            population=pop,
            fitness=fit,
            species=sp,
            alpha=0.4 * 1,  # 0.4
            gamma=-0.8 * 1,  # 0.8
            sigma=120  # 120
        ),
        termination_fn=lambda state, lim: termination_by_generation(
            state,
            lim
        )
    )

    gas2 = GeneticAlgorithmWithSpeciation(
        init_fn=lambda pop_size, dim: init_real_population(
            pop_size,
            dim,
            lower_bound=-100,
            upper_bound=100
        ),
        selection_fn=lambda pop, fit, size: tournament_selection(
            pop,
            fit,
            population_size=size,
            tournament_size=2,
            elite_size=math.ceil(size * 0.02)
        ),
        crossover_fn=lambda pop, fit: crossover_operator(
            pop,
            fit,
            proportion=0.6,
            crossover_fn=adaptive_blend_crossover_diversity,
            alpha_min=0.4,
            alpha_max=2.5
        ),
        mutation_fn=lambda pop, fit: mutation_operator(
            pop,
            fit,
            proportion=0.6,
            mutation_fn=adaptive_gaussian_mutation_diversity,
            sigma_min=0.6,
            sigma_max=2.5
        ),
        clusterization_fn=lambda pop:
        # dbscan_clusterization_indices(
        #    population=pop,
        #    eps=0.4,
        #    min_samples=5
        # ),
        hdbscan_clusterization_indices(
            population=pop,
            min_cluster_size=10
        ),
        # kmeans_clusterization_indices(
        #    population=pop,
        #    n_clusters=8
        # ),
        # optics_clusterization_indices(
        #    population=pop,
        #    xi=0.3,
        #    min_samples=5,
        #    min_cluster_size=10
        # ),
        role_assignment_fn=lambda sp: assign_roles_median(
            species=sp
        ),
        interaction_fn=lambda pop, fit,
                              sp: predator_prey_interaction_with_mean_fitness_update_and_scale(
            population=pop,
            fitness=fit,
            species=sp,
            alpha=0.3 * 1,  # 0.4
            gamma=-0.8 * 1,  # 0.8
            sigma=4  # 120
        ),
        termination_fn=lambda state, lim: termination_by_generation(
            state,
            lim
        )
    )

    gas3 = GeneticAlgorithmWithSpeciation(
        init_fn=lambda pop_size, dim: init_real_population(
            pop_size,
            dim,
            lower_bound=-100,
            upper_bound=100
        ),
        selection_fn=lambda pop, fit, size: tournament_selection(
            pop,
            fit,
            population_size=size,
            tournament_size=2,
            elite_size=math.ceil(size * 0.02)
        ),
        crossover_fn=lambda pop, fit: crossover_operator(
            pop,
            fit,
            proportion=0.6,
            crossover_fn=adaptive_blend_crossover_diversity,
            alpha_min=0.4,
            alpha_max=2.5
        ),
        mutation_fn=lambda pop, fit: mutation_operator(
            pop,
            fit,
            proportion=0.6,
            mutation_fn=adaptive_gaussian_mutation_diversity,
            sigma_min=0.6,
            sigma_max=2.5
        ),
        clusterization_fn=lambda pop:
        # dbscan_clusterization_indices(
        #    population=pop,
        #    eps=0.4,
        #    min_samples=5
        # ),
        hdbscan_clusterization_indices(
            population=pop,
            min_cluster_size=10
        ),
        # kmeans_clusterization_indices(
        #    population=pop,
        #    n_clusters=8
        # ),
        # optics_clusterization_indices(
        #    population=pop,
        #    xi=0.3,
        #    min_samples=5,
        #    min_cluster_size=10
        # ),
        role_assignment_fn=lambda sp: assign_roles_median(
            species=sp
        ),
        interaction_fn=lambda pop, fit,
                              sp: predator_prey_interaction_with_mean_fitness_update_with_solve_and_scale(
            population=pop,
            fitness=fit,
            species=sp,
            alpha=0.3 * 1,  # 0.4
            gamma=-0.8 * 1,  # 0.8
            sigma=4  # 120
        ),
        termination_fn=lambda state, lim: termination_by_generation(
            state,
            lim
        )
    )

    ga.run(population_size, dimension, limit, func, logger, species_logger_for_classic_ga)
    gas0.run(population_size, dimension, limit, func, logger, species_logger)
    # gas1.run()
    gas2.run(population_size, dimension, limit, func, logger, species_logger)
    species_logger.clear()
    gas3.run(population_size, dimension, limit, func, logger, species_logger)

    from visualization import show_landscape, species_analysis
    from utils.visualization import show_graph

    if dimension == 2:
        show_landscape(species_logger, func, predator_prey=False)

    show_graph(data_logger=logger, labels=[('GA', 1),
                                           ('GA with speciation classic', 1),
                                           # ('GA with speciation update', 1),
                                           ('GA with speciation update and scale', 1),
                                           ('GA with speciation update with solve and scale', 1)])

    # species_analysis(species_logger)

if 1 == 2:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    random.seed(241)
    np.random.seed(241)

    logger = DataLogger()
    species_logger_for_classic_ga = SpeciesLoggerForOneSpecies()
    species_logger = SpeciesLogger()

    population_size = 200
    dimension = 2
    limit = 1000

    ga = GeneticAlgorithm(
        population_size=population_size,
        objective=griewangks_function,
        init_fn=lambda pop_size: init_real_population(
            pop_size,
            dimension=dimension,
            lower_bound=-100,
            upper_bound=100
        ),
        selection_fn=lambda pop, fit, size: tournament_selection(
            pop,
            fit,
            population_size=size,
            tournament_size=2,
            elite_size=2
        ),
        crossover_fn=lambda pop: crossover_operator(
            pop,
            proportion=0.5,
            crossover_fn=blend_crossover,
            alpha=0.5
        ),
        mutation_fn=lambda pop: mutation_operator(
            pop,
            proportion=0.5,
            mutation_fn=gaussian_mutation,
            sigma=0.1
        ),
        termination_fn=lambda state: termination_by_generation(
            state,
            limit=limit
        ),
        logger=logger,
        species_logger=species_logger_for_classic_ga
    )

    gas = GeneticAlgorithmWithSpeciation(
        population_size=population_size,
        objective=griewangks_function,
        init_fn=lambda pop_size: init_real_population(
            pop_size,
            dimension=dimension,
            lower_bound=-100,
            upper_bound=100
        ),
        selection_fn=lambda pop, fit, size: tournament_selection(
            pop,
            fit,
            population_size=size,
            tournament_size=2,
            elite_size=int(size * 0.02)
        ),
        crossover_fn=lambda pop: crossover_operator(
            pop,
            proportion=0.6,
            crossover_fn=blend_crossover,
            alpha=0.8
        ),
        mutation_fn=lambda pop: mutation_operator(
            pop,
            proportion=0.6,
            mutation_fn=gaussian_mutation,
            sigma=0.8
        ),
        clusterization_fn=lambda pop:
        # hdbscan_clusterization_indices(
        #    population=pop,
        #    min_cluster_size=10
        # ),
        dbscan_clusterization_indices(
            population=pop,
            eps=0.4,
            min_samples=5
        ),
        # kmeans_clusterization_indices(
        #    population=pop,
        #    n_clusters=8
        # ),
        # optics_clusterization_indices(
        #    population=pop,
        #    xi=0.3,
        #    min_samples=5,
        #    min_cluster_size=10
        # ),
        role_assignment_fn=lambda sp: assign_predator_prey_by_median(
            species=sp
        ),
        interaction_fn=lambda pop, fit, sp: predator_prey_interaction(
            population=pop,
            fitness=fit,
            species=sp,
            alpha=0.4 * 1,
            gamma=-0.8 * 1,
            sigma=120
        ),
        termination_fn=lambda state: termination_by_generation(
            state,
            limit=limit
        ),
        logger=logger,
        species_logger=species_logger
    )

    ga.run()
    gas.run()

    from visualization import show_graph, show_landscape

    show_landscape(species_logger, griewangks_function, predator_prey=False)

    show_graph(data_logger=logger, labels=[('GA', 1), ('GA with speciation', 1)])
