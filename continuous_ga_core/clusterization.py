import numpy as np
from typing import Optional


def _assign_noise_hybrid(
        points: np.ndarray,
        clusters: list[np.ndarray],
        noise_indices: np.ndarray,
        distance_threshold: float | None = None,
        fallback_n_clusters: int | None = None,
        random_state: int = 42
) -> list[np.ndarray]:
    """
    Гибридное устранение шума.

    Если первичный алгоритм не нашёл ни одного кластера,
    применяется fallback-кластеризация через KMeans.
    """
    from sklearn.cluster import KMeans

    # ===== СЛУЧАЙ 1: кластеры отсутствуют вообще =====
    if len(clusters) == 0:
        n_samples = points.shape[0]

        if fallback_n_clusters is None:
            # Эвристика: sqrt(n / 2), минимум 1
            fallback_n_clusters = 10  # max(1, int(np.sqrt(n_samples / 2)))

        kmeans = KMeans(
            n_clusters=fallback_n_clusters,
            random_state=random_state,
            n_init="auto"
        )
        labels = kmeans.fit_predict(points)

        return [np.where(labels == i)[0] for i in range(fallback_n_clusters)]

    # ===== СЛУЧАЙ 2: есть кластеры, есть шум =====
    cluster_centers = np.vstack([points[indices].mean(axis=0) for indices in clusters])

    new_clusters = [indices.copy() for indices in clusters]
    remaining_noise = []

    for idx in noise_indices:
        point = points[idx]
        distances = np.linalg.norm(cluster_centers - point, axis=1)
        nearest_cluster = np.argmin(distances)

        if distance_threshold is None or distances[nearest_cluster] <= distance_threshold:
            new_clusters[nearest_cluster] = np.append(new_clusters[nearest_cluster], idx)
        else:
            remaining_noise.append(idx)

    # ===== СЛУЧАЙ 3: часть шума осталась =====
    if remaining_noise:
        noise_points = points[remaining_noise]
        kmeans = KMeans(
            n_clusters=len(new_clusters),
            random_state=random_state,
            n_init="auto"
        )
        labels = kmeans.fit_predict(noise_points)

        for label, idx in zip(labels, remaining_noise):
            new_clusters[label] = np.append(new_clusters[label], idx)

    return new_clusters


def _prepare_data(population: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    if population.ndim == 1:
        population = population.reshape(-1, 1)
    return StandardScaler().fit_transform(population)


def _labels_to_clusters(
        points: np.ndarray,
        labels: np.ndarray
) -> list[np.ndarray]:
    clusters = []
    noise = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if label == -1:
            noise.extend(indices)
        else:
            clusters.append(indices)
    if noise:
        clusters = _assign_noise_hybrid(points, clusters, np.array(noise))
    return clusters


def kmeans_clusterization_indices(
        population: np.ndarray,
        n_clusters: int
) -> list[np.ndarray]:
    """
    Возвращает список массивов, содеражщих индексы индиводов каждого кластера
    """
    from sklearn.cluster import KMeans

    X = _prepare_data(population)
    labels = KMeans(n_clusters=n_clusters).fit_predict(X)

    return _labels_to_clusters(X, labels)


def dbscan_clusterization_indices(
        population: np.ndarray,
        eps: float,
        min_samples: int
) -> list[np.ndarray]:
    """
    Возвращает список массивов, содеражщих индексы индиводов каждого кластера
    """
    from sklearn.cluster import DBSCAN

    X = _prepare_data(population)
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit_predict(X)

    return _labels_to_clusters(X, labels)


def hdbscan_clusterization_indices(
        population: np.ndarray,
        min_cluster_size: int = 10
) -> list[np.ndarray]:
    """
    Возвращает список массивов, содеражщих индексы индиводов каждого кластера
    """
    from sklearn.cluster import HDBSCAN

    X = _prepare_data(population)
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size
    ).fit_predict(X)

    return _labels_to_clusters(X, labels)


def optics_clusterization_indices(
        population: np.ndarray,
        min_samples: int,
        xi: float,
        min_cluster_size: int
) -> list[np.ndarray]:
    """
    Возвращает список массивов, содеражщих индексы индиводов каждого кластера
    """
    from sklearn.cluster import OPTICS

    X = _prepare_data(population)
    labels = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size
    ).fit_predict(X)

    return _labels_to_clusters(X, labels)


def agglomerative_clusterization_indices(
        population: np.ndarray,
        n_clusters: int,
        linkage: str = "ward"
) -> list[np.ndarray]:
    """
    Возвращает список массивов, содеражщих индексы индиводов каждого кластера
    """
    from sklearn.cluster import AgglomerativeClustering

    X = _prepare_data(population)

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    ).fit_predict(X)

    return _labels_to_clusters(X, labels)


def whitshart_clusterization_indices(
        population: np.ndarray
) -> list[np.ndarray]:
    """
    Возвращает список массивов, содеражщих индексы индиводов каждого кластера
    """
    raise RuntimeError("Not implemented Whitshart clusterization")


def _plot_2d_clusters(
        points: np.ndarray,
        clusters: list[np.ndarray],
        noise_indices: np.ndarray | None = None,
        title: str | None = None
) -> None:
    """
    Визуализирует 2D точки, сгруппированные в кластеры.

    Parameters
    ----------
    points : np.ndarray
        Массив формы (n_samples, 2)
    clusters : list[np.ndarray]
        Список массивов индексов кластеров
    noise_indices : np.ndarray | None
        Индексы шума (label == -1)
    title : str | None
        Заголовок графика
    """
    import matplotlib.pyplot as plt

    if points.shape[1] != 2:
        raise ValueError("points must have shape (n_samples, 2)")

    plt.figure(figsize=(8, 6))

    for i, indices in enumerate(clusters):
        plt.scatter(
            points[indices, 0],
            points[indices, 1],
            label=f"Cluster {i}",
            s=40
        )

    if noise_indices is not None and len(noise_indices) > 0:
        plt.scatter(
            points[noise_indices, 0],
            points[noise_indices, 1],
            marker="x",
            label="Noise"
        )

    plt.legend()
    plt.xlabel("X1")
    plt.ylabel("X2")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()


def _plot_multiple_2d_clusters_subplots(
        datasets: list[tuple[np.ndarray, list[np.ndarray], Optional[np.ndarray]]],
        titles: Optional[list[str]] = None
) -> None:
    """
    Визуализирует несколько наборов 2D точек с кластерами в виде подграфиков

    Parameters
    ----------
    datasets : List[Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]]
        Список кортежей вида (points, clusters, noise_indices):
            - points: массив формы (n_samples, 2)
            - clusters: список массивов индексов кластеров
            - noise_indices: массив индексов шума (label == -1) или None
    titles : List[str] | None
        Список заголовков для каждого графика
    """
    import matplotlib.pyplot as plt
    import math

    n_plots = len(datasets)
    if titles is not None and len(titles) != n_plots:
        raise ValueError("Длина titles должна совпадать с количеством наборов данных")

    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, (points, clusters, noise_indices) in enumerate(datasets):
        ax = axes[idx]

        if points.shape[1] != 2:
            raise ValueError(f"points в наборе {idx} должны иметь форму (n_samples, 2)")

        for i, indices in enumerate(clusters):
            ax.scatter(
                points[indices, 0],
                points[indices, 1],
                label=f"Cluster {i}",
                s=40
            )

        if noise_indices is not None and len(noise_indices) > 0:
            ax.scatter(
                points[noise_indices, 0],
                points[noise_indices, 1],
                marker="x",
                label="Noise"
            )

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        if titles is not None:
            ax.set_title(titles[idx])
        ax.legend()

    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def _plot_points(X: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets import (
        make_blobs,
        make_classification,
        make_gaussian_quantiles,
        make_moons,
        make_circles,
        make_swiss_roll,
        make_s_curve
    )

    X0, _ = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.2,
        flip_y=0.05,
        random_state=42
    )

    X1, _ = make_blobs(
        n_samples=500,
        centers=4,
        cluster_std=[1.0, 2.5, 0.5, 1.5],
        random_state=42
    )

    X2, _ = make_gaussian_quantiles(
        n_samples=500,
        n_features=2,
        n_classes=3,
        random_state=42
    )

    X3, _ = make_moons(
        n_samples=500,
        noise=0.1,
        random_state=42
    )

    X4, _ = make_circles(
        n_samples=500,
        noise=0.05,
        factor=0.5,
        random_state=42
    )

    X5, _ = make_swiss_roll(
        n_samples=500,
        noise=0.1,
        random_state=42
    )
    X5 = X5[:, [0, 2]]

    X6, _ = make_s_curve(
        n_samples=1000,
        noise=0.1,
        random_state=42
    )
    X6 = X6[:, [0, 2]]

    XS = [X0, X1, X2, X3, X4, X6]
    datasets = []

    for X in XS:
        # clusters = kmeans_clusterization_indices(X, n_clusters=4)
        # clusters = dbscan_clusterization_indices(X, eps=0.25, min_samples=5)
        clusters = hdbscan_clusterization_indices(X, min_cluster_size=10)
        # clusters = optics_clusterization_indices(X, min_samples=4, xi=0.15, min_cluster_size=10)
        # clusters = agglomerative_clusterization_indices(X, n_clusters=4)

        datasets.append((X, clusters, None))

    _plot_multiple_2d_clusters_subplots(
        datasets,
        titles=["make_classification", "make_blobs", "make_gaussian_quantiles",
                "make_moons", "make_circles", "make_s_curve"]
    )
