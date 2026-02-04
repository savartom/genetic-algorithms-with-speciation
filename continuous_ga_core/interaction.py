import math
import random
import numpy as np
from continuous_ga_core.population import SpeciesState


def evolve_species(
        population: np.ndarray,
        fitness: np.ndarray,
        species: SpeciesState,
        crossover_fn,
        mutation_fn
) -> np.ndarray:
    individuals = population[species.indices]
    fitness_individuals = fitness[species.indices]

    offsprings_c = crossover_fn(individuals, fitness_individuals)
    offsprings_m = mutation_fn(individuals, fitness_individuals)

    sum_size = len(offsprings_c) + len(offsprings_m)
    species.indices = (
        np.append(species.indices, np.arange(len(population), len(population) + sum_size))
    )
    return np.vstack([population, offsprings_c, offsprings_m])


def recalc_params(
        population: np.ndarray,
        fitness: np.ndarray,
        species: SpeciesState
):
    idx = species.indices
    species.mean_individual = population[idx].mean(axis=0)
    species.mean_fitness = fitness[idx].mean()


def assign_roles_median(species: list[SpeciesState]) -> None:
    median = np.median([sp.mean_fitness for sp in species])
    for sp in species:
        sp.role = 'prey' if sp.mean_fitness < median else 'predator'


def assign_roles_power(species: list[SpeciesState], alpha=2.5) -> None:
    for sp in species:
        sp.role = 'prey' if pow(sp.mean_fitness, alpha) < random.random() else 'predator'


def assign_roles_random(species: list[SpeciesState]) -> None:
    for sp in species:
        sp.role = 'prey' if random.random() > 0.5 else 'predator'


def assign_roles_inverse_size(species: list[SpeciesState]) -> None:
    s = len(species)
    if s == 0:
        return

    N = sum(sp.size for sp in species)

    weights = [(N / (sp.size * s)) for sp in species]

    max_w = max(weights)
    if max_w == 0:
        probs = [0.0] * s
    else:
        probs = [w / max_w for w in weights]

    for sp, p_predator in zip(species, probs):
        sp.role = 'predator' if random.random() < p_predator else 'prey'


def assign_roles_inverse_size_with_gamma(species: list[SpeciesState], gamma: float = 1.0) -> None:
    s = len(species)
    N = sum(sp.size for sp in species)

    for sp in species:
        intensity = (N / (sp.size * s)) * gamma
        p_predator = 1.0 - np.exp(-intensity)
        sp.role = 'predator' if random.random() < p_predator else 'prey'


def assign_roles_inverse_size_softmax(species: list[SpeciesState], gamma: float = 1.0) -> None:
    logits = np.array([-gamma * sp.size for sp in species])
    probs = np.exp(logits) / np.sum(np.exp(logits))

    for sp, p in zip(species, probs):
        sp.role = 'predator' if random.random() < p else 'prey'


def assign_roles_median_stochastic(
        species: list[SpeciesState],
        alpha: float = 3.0
) -> None:
    species_sorted = sorted(species, key=lambda sp: sp.mean_fitness, reverse=True)
    n = len(species_sorted)

    worst = species_sorted[0]
    best = species_sorted[-1]

    worst.role = 'predator'
    best.role = 'prey'
    if n <= 2:
        return

    median_idx = n // 2
    for i, sp in enumerate(species_sorted[1:-1], start=1):
        d = (i - median_idx) / (n / 2)
        p_prey = 1 / (1 + abs(d) ** alpha)
        sp.role = 'prey' if random.random() < p_prey else 'predator'


def predator_prey_interaction(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha: float = 0.5,
        gamma: float = 0.8,
        sigma: float = 10.0
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :], axis=2
        )  # (K, S)

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness


def predator_prey_interaction_with_mean_fitness_update(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha: float = 0.5,
        gamma: float = 0.8,
        sigma: float = 10.0
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # Взаимодействие средних fitness видов
    same_role = roles[:, None] == roles[None, :]
    is_predator_i = roles[:, None] == 'predator'
    coefficients = (
            -alpha * same_role
            - gamma * (~same_role & is_predator_i)
            + gamma * (~same_role & ~is_predator_i)
    )
    np.fill_diagonal(coefficients, 0.0)
    mean_fitness = mean_fitness + 0.5 * coefficients @ mean_fitness

    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :], axis=2
        )  # (K, S)

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness


def predator_prey_interaction_with_mean_fitness_update_and_scale(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha: float = 0.5,
        gamma: float = 0.8,
        sigma: float = 10.0,
        eps: float = 1e-12
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S,)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # --------------------------------------------------
    # 1. Глобальный масштаб
    # --------------------------------------------------
    global_dists = np.linalg.norm(
        population[:, None, :] - means_individual[None, :, :],
        axis=2
    )
    global_scale = np.median(global_dists) + eps

    # --------------------------------------------------
    # 2. Взаимодействие средних fitness видов
    # --------------------------------------------------
    same_role = roles[:, None] == roles[None, :]
    is_predator_i = roles[:, None] == 'predator'

    coefficients = (
            -alpha * same_role
            - gamma * (~same_role & is_predator_i)
            + gamma * (~same_role & ~is_predator_i)
    )
    np.fill_diagonal(coefficients, 1.0)
    mean_fitness = mean_fitness + 0.5 * coefficients @ mean_fitness

    # --------------------------------------------------
    # 3. Основной цикл по видам
    # --------------------------------------------------
    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :],
            axis=2
        )  # (K, S)

        dists /= global_scale

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness


def predator_prey_interaction_with_scale(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha: float = 0.5,
        gamma: float = 0.8,
        sigma: float = 10.0,
        eps: float = 1e-12
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S,)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # --------------------------------------------------
    # 1. Глобальный масштаб
    # --------------------------------------------------
    global_dists = np.linalg.norm(
        population[:, None, :] - means_individual[None, :, :],
        axis=2
    )
    global_scale = np.median(global_dists) + eps

    # --------------------------------------------------
    # 2. Основной цикл по видам
    # --------------------------------------------------
    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :],
            axis=2
        )  # (K, S)

        dists /= global_scale

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness


def adaptive_predator_prey_interaction_with_scale(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        ratio: float,
        alpha_min: float = -0.6,
        alpha_max: float = -0.4,
        gamma_min: float = 0.4,
        gamma_max: float = 0.6,
        sigma: float = 10.0,
        eps: float = 1e-12
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S,)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # --------------------------------------------------
    # 1. Глобальный масштаб
    # --------------------------------------------------
    global_dists = np.linalg.norm(
        population[:, None, :] - means_individual[None, :, :],
        axis=2
    )
    global_scale = np.median(global_dists) + eps

    # --------------------------------------------------
    # 2. Вычисление alpha и gamma
    # --------------------------------------------------
    alpha = alpha_min + (alpha_max - alpha_min) * np.exp(-3 * ratio)
    gamma = gamma_min + (gamma_max - gamma_min) * np.exp(-3 * ratio)

    # --------------------------------------------------
    # 3. Основной цикл по видам
    # --------------------------------------------------
    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :],
            axis=2
        )  # (K, S)

        dists /= global_scale

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness


from scipy.spatial.distance import pdist


def diversity_distance_uniformity(X: np.ndarray):
    if len(X) < 2:
        return 0.0

    dists = pdist(X)
    mean = np.mean(dists)

    if mean == 0:
        return 0.0

    cv = np.std(dists) / mean
    return np.clip(1 - cv, 0.0, 1.0)


def adaptive_predator_prey_interaction_with_scale_bad(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha_min: float = -0.6,
        alpha_max: float = -0.4,
        gamma_min: float = 0.4,
        gamma_max: float = 0.6,
        sigma: float = 10.0,
        eps: float = 1e-12
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S,)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # --------------------------------------------------
    # 1. Глобальный масштаб
    # --------------------------------------------------
    global_dists = np.linalg.norm(
        population[:, None, :] - means_individual[None, :, :],
        axis=2
    )
    global_scale = np.median(global_dists) + eps

    # --------------------------------------------------
    # 2. Вычисление alpha и gamma
    # --------------------------------------------------
    diversity = diversity_distance_uniformity(population)
    alpha = alpha_min + (alpha_max - alpha_min) * diversity
    gamma = gamma_min + (gamma_max - gamma_min) * diversity

    #print(diversity)

    # --------------------------------------------------
    # 3. Основной цикл по видам
    # --------------------------------------------------
    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :],
            axis=2
        )  # (K, S)

        dists /= global_scale

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness


def predator_prey_interaction_with_scale_bad_adap(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha: float = -0.5,
        alpha_tau: float = 1 / math.sqrt(2.0 * 100),
        gamma: float = 0.5,
        gamma_tau: float = 1 / math.sqrt(2.0 * 100),
        alpha_min: float = -0.6,
        alpha_max: float = -0.4,
        gamma_min: float = 0.4,
        gamma_max: float = 0.6,
        sigma: float = 0.8,
        eps: float = 1e-12
):
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S,)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # --------------------------------------------------
    # 1. Глобальный масштаб
    # --------------------------------------------------
    global_dists = np.linalg.norm(
        population[:, None, :] - means_individual[None, :, :],
        axis=2
    )
    global_scale = np.median(global_dists) + eps

    # --------------------------------------------------
    # 2. Обновление alpha и gamma
    # --------------------------------------------------
    alpha = alpha * np.exp(alpha_tau * np.random.random())
    alpha = max(min(alpha, alpha_max), alpha_min)

    gamma = gamma * np.exp(gamma_tau * np.random.random())
    gamma = max(min(gamma, gamma_max), gamma_min)

    # --------------------------------------------------
    # 3. Основной цикл по видам
    # --------------------------------------------------
    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :],
            axis=2
        )  # (K, S)

        dists /= global_scale

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness, alpha, gamma


def predator_prey_interaction_with_mean_fitness_update_with_solve_and_scale(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha: float = 0.5,
        gamma: float = 0.8,
        sigma: float = 10.0,
        eps: float = 1e-12
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S,)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # --------------------------------------------------
    # 1. Глобальный масштаб
    # --------------------------------------------------
    global_dists = np.linalg.norm(
        population[:, None, :] - means_individual[None, :, :],
        axis=2
    )
    global_scale = np.median(global_dists) + eps

    # --------------------------------------------------
    # 2. Взаимодействие средних fitness видов
    # --------------------------------------------------
    same_role = roles[:, None] == roles[None, :]
    is_predator_i = roles[:, None] == 'predator'

    coefficients = (
            -alpha * same_role
            - gamma * (~same_role & is_predator_i)
            + gamma * (~same_role & ~is_predator_i)
    )
    np.fill_diagonal(coefficients, 0.0)
    I = np.eye(coefficients.shape[0])
    A = I - coefficients
    mean_fitness = np.linalg.solve(A + eps * np.eye(A.shape[0]), mean_fitness)

    # --------------------------------------------------
    # 3. Основной цикл по видам
    # --------------------------------------------------
    for i, sp in enumerate(species):
        individuals = population[sp.indices]  # (K, D)

        dists = np.linalg.norm(
            individuals[:, None, :] - means_individual[None, :, :],
            axis=2
        )  # (K, S)

        dists /= global_scale

        influence = np.maximum(0, sigma - dists) * mean_fitness

        same_role = (roles == sp.role)
        influence[:, same_role] *= -alpha
        influence[:, ~same_role] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1) * (0.25 if idx_min_fitness in sp.indices else 1.0)

        new_fitness[sp.indices] += delta

    return new_fitness


def predator_prey_interaction_with_mean_fitness_update_and_scale_and_quantile(
        population: np.ndarray,
        fitness: np.ndarray,
        species: list[SpeciesState],
        alpha: float = 0.3,
        gamma: float = 0.8,
        sigma: float = 4.0,
        q: float = 0.2,
        eps: float = 1e-12
) -> np.ndarray:
    new_fitness = fitness.copy()

    means_individual = np.array([sp.mean_individual for sp in species])  # (S, D)
    mean_fitness = np.array([sp.mean_fitness for sp in species])  # (S,)
    roles = np.array([sp.role for sp in species])

    idx_min_fitness = fitness.argmin()

    # --------------------------------------------------
    # 1. Глобальный масштаб
    # --------------------------------------------------
    global_dists = np.linalg.norm(
        population[:, None, :] - means_individual[None, :, :],
        axis=2
    )
    global_scale = np.median(global_dists) + eps

    # --------------------------------------------------
    # 2. Взаимодействие средних fitness видов
    # --------------------------------------------------
    same_role = roles[:, None] == roles[None, :]
    is_predator_i = roles[:, None] == 'predator'

    coefficients = (
            -alpha * same_role
            - gamma * (~same_role & is_predator_i)
            + gamma * (~same_role & ~is_predator_i)
    )
    np.fill_diagonal(coefficients, 1.0)
    mean_fitness = mean_fitness + 0.5 * coefficients @ mean_fitness

    # --------------------------------------------------
    # 3. ВСЕ попарные расстояния между особями
    # --------------------------------------------------
    all_pairwise_dists = np.linalg.norm(
        population[:, None, :] - population[None, :, :],
        axis=2
    )

    # --------------------------------------------------
    # 4. Основной цикл по видам
    # --------------------------------------------------
    for i, sp in enumerate(species):
        idx_i = sp.indices
        individuals_count = len(idx_i)

        dists = np.empty((individuals_count, len(species)))

        for j, sp_j in enumerate(species):
            idx_j = sp_j.indices

            pairwise = all_pairwise_dists[np.ix_(idx_i, idx_j)]

            dists[:, j] = np.quantile(pairwise, q, axis=1)

        dists /= global_scale

        influence = np.maximum(0.0, sigma - dists) * mean_fitness

        same_role_mask = (roles == sp.role)
        influence[:, same_role_mask] *= -alpha
        influence[:, ~same_role_mask] *= -gamma if sp.role == 'predator' else gamma

        delta = influence.sum(axis=1)
        if idx_min_fitness in idx_i:
            delta *= 0.25

        new_fitness[idx_i] += delta

    return new_fitness


if __name__ == "__main__":
    ...
