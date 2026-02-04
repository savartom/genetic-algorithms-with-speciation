import plotly.express as px
import umap
from scipy.ndimage import minimum_filter
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.stats import hmean
import plotly.graph_objects as go

from continuous_ga_core.logger import SpeciesLogger


def show_species(species_logger):
    generations = species_logger.generations
    if species_logger.generations[0][0][0].shape[0] != 2:
        generations = reduce_generations(generations)

    group_names = [[(len(species), hmean(species), min(species)) for species in generation]
                   for generation in species_logger.fitness]

    color_pool = px.colors.qualitative.Dark24 * 10

    # Выравниваем число trace-ов до максимального числа групп в любом поколении
    max_groups = max(len(gen) for gen in generations)
    num_generations = len(generations)

    # ---------------------------
    # Фигура и фреймы
    # ---------------------------
    fig = go.Figure()
    frames = []

    # Построим фреймы, каждый фрейм содержит ровно max_groups trace-ов.
    # Если в поколении меньше групп — добавим пустые (невидимые) trace-ы,
    # чтобы индексы trace-ов совпадали между поколениями.
    for gen_idx, groups in enumerate(generations):
        frame_traces = []
        for slot in range(max_groups):
            if slot < len(groups):
                points = groups[slot]
                # Явно формируем x и y как просили
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                name = ""
                fitness_values = species_logger.fitness[gen_idx][slot]
                frame_traces.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        marker=dict(size=7, color=color_pool[slot]),
                        name=name,
                        visible=True,
                        showlegend=False
                    )
                )
            else:
                # Пустой невидимый trace — гарантирует постоянство индексов trace-ов
                frame_traces.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        marker=dict(size=7, color=color_pool[slot]),
                        name="",
                        visible=False,
                        showlegend=False
                    )
                )

        # Аннотация (номер поколения, число групп, имена)
        annotation_text = (
                f"<b>Generation: {gen_idx + 1}</b><br>"
                f"Number of groups: {len(groups)}<br>" +
                "<br>".join([f"{i}) size {size}, mean: {ind:.3f}, min: {m:.3f}"
                             for i, (size, ind, m) in enumerate(group_names[gen_idx], start=1)])
        )

        frames.append(go.Frame(
            data=frame_traces,
            name=f"gen{gen_idx}",
            layout=go.Layout(
                annotations=[dict(
                    text=annotation_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=1.05, y=0.95,
                    align="left",
                    xanchor="left",
                    yanchor="top"
                )]
            )
        ))

    # ---------------------------
    # Начальное состояние: добавляем traces из первого фрейма (ровно max_groups)
    # ---------------------------
    fig.add_traces(frames[0].data)
    fig.update_layout(annotations=frames[0].layout.annotations)

    # Устанавливаем frames
    fig.frames = frames

    # ---------------------------
    # Слайдер: шаги — по одному фрейму, метки — номера поколений
    # ---------------------------
    steps = []
    for i, fr in enumerate(frames):
        steps.append(dict(
            method="animate",
            args=[
                [fr.name],
                {"mode": "immediate",
                 "frame": {"duration": 0, "redraw": False},
                 "transition": {"duration": 0}}
            ],
            label=str(i)
        ))

    fig.update_layout(
        sliders=[dict(
            active=0,
            steps=steps,
            x=0.1,
            y=-0.12,
            len=0.8
        )],
        width=800,
        height=600,
        title="Speciation in the genetic algorithm",
        xaxis_title="X",
        yaxis_title="Y",
        margin=dict(r=300)  # место справа для аннотации
    )

    fig.show()


def show_changes(species_logger):
    generations = species_logger.generations
    if species_logger.generations[0][0][0].shape[0] != 2:
        generations = reduce_generations(generations)

    group_names = [[(len(species), hmean(species), min(species)) for species in generation]
                   for generation in species_logger.fitness]

    color_pool = px.colors.qualitative.Dark24 * 10

    # Выравниваем число trace-ов до максимального числа групп в любом поколении
    max_groups = max(len(gen) for gen in generations)
    num_generations = len(generations)

    # ---------------------------
    # Фигура и фреймы
    # ---------------------------
    fig = go.Figure()
    frames = []

    # Построим фреймы, каждый фрейм содержит ровно max_groups trace-ов.
    # Если в поколении меньше групп — добавим пустые (невидимые) trace-ы,
    # чтобы индексы trace-ов совпадали между поколениями.
    for gen_idx, groups in enumerate(generations):
        frame_traces = []
        for slot in range(max_groups):
            if slot < len(groups):
                points = groups[slot]
                # Явно формируем x и y как просили
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                name = ""
                fitness_values = species_logger.fitness[gen_idx][slot]
                frame_traces.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        marker=dict(
                            size=7,
                            color=fitness_values,
                            colorscale="Viridis",
                            showscale=False,
                            line=dict(
                                color=color_pool[slot],  # цвет контура
                                width=1
                            )
                        ),
                        name=name,
                        visible=True,
                        showlegend=False
                    )
                )
            else:
                # Пустой невидимый trace — гарантирует постоянство индексов trace-ов
                frame_traces.append(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        marker=dict(size=7, color=color_pool[slot]),
                        name="",
                        visible=False,
                        showlegend=False
                    )
                )

        # Аннотация (номер поколения, число групп, имена)
        annotation_text = (
                f"<b>Generation: {gen_idx + 1}</b><br>"
                f"Number of groups: {len(groups)}<br>" +
                "<br>".join([f"{i}) size {size}, hmean: {ind:.3f}, min: {m:.3f}"
                             for i, (size, ind, m) in enumerate(group_names[gen_idx], start=1)])
        )

        frames.append(go.Frame(
            data=frame_traces,
            name=f"gen{gen_idx}",
            layout=go.Layout(
                annotations=[dict(
                    text=annotation_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=1.05, y=0.95,
                    align="left",
                    xanchor="left",
                    yanchor="top"
                )]
            )
        ))

    # ---------------------------
    # Начальное состояние: добавляем traces из первого фрейма (ровно max_groups)
    # ---------------------------
    fig.add_traces(frames[0].data)
    fig.update_layout(annotations=frames[0].layout.annotations)

    # Устанавливаем frames
    fig.frames = frames

    # ---------------------------
    # Слайдер: шаги — по одному фрейму, метки — номера поколений
    # ---------------------------
    steps = []
    for i, fr in enumerate(frames):
        steps.append(dict(
            method="animate",
            args=[
                [fr.name],
                {"mode": "immediate",
                 "frame": {"duration": 0, "redraw": True},
                 "transition": {"duration": 0}}
            ],
            label=str(i)
        ))

    fig.update_layout(
        sliders=[dict(
            active=0,
            steps=steps,
            x=0.1,
            y=-0.12,
            len=0.8
        )],
        width=800,
        height=600,
        title="Speciation in the genetic algorithm",
        xaxis_title="X",
        yaxis_title="Y",
        margin=dict(r=300)  # место справа для аннотации
    )

    fig.show()


def show_landscape(species_logger, func, num_in_linspace=1000, has_color=True, predator_prey=False):
    from scipy.stats import hmean
    import io
    import base64

    name_color = 'darkgrey'
    predator_color = 'orange'
    prey_color = 'green'

    generations = species_logger.generations
    roles = species_logger.roles
    if generations[0][0][0].shape[0] != 2:
        generations = reduce_generations(generations)

    group_names = [[(len(species), hmean(species), min(species)) for species in generation]
                   for generation in species_logger.fitness]

    color_pool = [col for col in px.colors.qualitative.Dark24
                  if col not in {'#FB0D0D', '#FC0080', '#AF0038'}] * 10
    max_groups = max(len(gen) for gen in generations)

    # ---------------------------
    # Подготовка данных для фонового ландшафта
    # ---------------------------
    all_points, all_values = [], []
    for gen_idx, groups in enumerate(generations):
        for group_idx, points in enumerate(groups):
            fitness_vals = species_logger.fitness[gen_idx][group_idx]
            for p, f in zip(points, fitness_vals):
                all_points.append([p[0], p[1]])
                all_values.append(f)

    all_points = np.array(all_points)
    all_values = np.array(all_values)

    # KNN для аппроксимации
    # knn = KNeighborsRegressor(n_neighbors=12, weights='distance')
    # knn.fit(all_points, all_values)

    # Сетка
    #min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_x, max_x = -100.0, 100.0
    #min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
    min_y, max_y = -100.0, 100.0
    gx = np.linspace(min_x, max_x, num_in_linspace)
    gy = np.linspace(min_y, max_y, num_in_linspace)
    GX, GY = np.meshgrid(gx, gy)
    # grid_pred = knn.predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    grid_pred = np.apply_along_axis(
        lambda p: func(p),
        axis=2,
        arr=np.dstack((GX, GY))
    )

    # ---------------------------
    # Поиск локальных минимумов на ландшафте
    # ---------------------------

    local_min_mask = (grid_pred == minimum_filter(grid_pred, size=5, mode='reflect'))

    # Убираем границы
    local_min_mask[:3, :] = False
    local_min_mask[-3:, :] = False
    local_min_mask[:, :3] = False
    local_min_mask[:, -3:] = False

    min_x_coords = GX[local_min_mask]
    min_y_coords = GY[local_min_mask]
    min_values = grid_pred[local_min_mask]

    # Жёсткая фильтрация по значению (оставляем только "глубокие" минимумы)
    value_threshold = np.percentile(grid_pred, 5)
    mask = min_values <= value_threshold

    min_points = np.c_[min_x_coords[mask], min_y_coords[mask]]

    # ---------------------------
    # Пространственная агрегация минимумов
    # ---------------------------

    # Объединяем близкие минимумы
    clustering = DBSCAN(
        eps=0.02 * max(max_x - min_x, max_y - min_y),
        min_samples=1
    ).fit(min_points)

    labels = clustering.labels_

    # Берём центр каждого кластера
    unique_labels = np.unique(labels)
    minima_x, minima_y = [], []

    for lbl in unique_labels:
        cluster_pts = min_points[labels == lbl]
        minima_x.append(cluster_pts[:, 0].mean())
        minima_y.append(cluster_pts[:, 1].mean())

    minima_z = [func(np.array([x, y])) for x, y in zip(minima_x, minima_y)]

    # Преобразуем в изображение с цветовой картой Viridis
    import matplotlib.pyplot as plt
    fig_img, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(grid_pred, cmap='viridis', origin='lower', extent=[min_x, max_x, min_y, max_y])

    buf = io.BytesIO()
    fig_img.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig_img)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()

    img_uri = "data:image/png;base64," + img_base64

    # ---------------------------
    # Создаём фигуру
    # ---------------------------
    fig = go.Figure()

    # Добавляем фон как изображение
    fig.add_layout_image(
        dict(
            source=img_uri,
            xref="x", yref="y",
            x=min_x, y=max_y,
            sizex=max_x - min_x, sizey=max_y - min_y,
            sizing="stretch",
            layer="below",
            opacity=1.0
        )
    )

    # ---------------------------
    # Локальные минимумы (красные точки)
    # ---------------------------
    minima_trace = go.Scatter(
        x=minima_x,
        y=minima_y,
        mode='markers',
        customdata=minima_z,
        hovertemplate=(
            "x: %{x:.4f}<br>"
            "y: %{y:.4f}<br>"
            "f(x,y): %{customdata:.6f}"
            "<extra></extra>"
        ),
        marker=dict(
            size=6,
            color='red',
            symbol='x'
        ),
        opacity=0.5,
        showlegend=False
    )

    # ---------------------------
    # Поиск локальных максимумов рядом с минимумами
    # ---------------------------

    from scipy.ndimage import maximum_filter

    # Маска локальных максимумов
    local_max_mask = (grid_pred == maximum_filter(grid_pred, size=5, mode='reflect'))

    # Убираем границы
    local_max_mask[:3, :] = False
    local_max_mask[-3:, :] = False
    local_max_mask[:, :3] = False
    local_max_mask[:, -3:] = False

    max_x_coords = GX[local_max_mask]
    max_y_coords = GY[local_max_mask]
    max_values = grid_pred[local_max_mask]

    max_points = np.c_[max_x_coords, max_y_coords]

    # --- Фильтрация: оставляем максимумы ТОЛЬКО рядом с минимумами ---
    from sklearn.neighbors import KDTree

    minima_points = np.c_[minima_x, minima_y]
    tree = KDTree(minima_points)

    # радиус поиска — тот же масштаб, что и у DBSCAN
    radius = 0.05 * max(max_x - min_x, max_y - min_y)

    idx = tree.query_radius(max_points, r=radius)
    near_min_mask = np.array([len(i) > 0 for i in idx])

    max_points = max_points[near_min_mask]
    max_values = max_values[near_min_mask]

    # ---------------------------
    # Пространственная агрегация максимумов
    # ---------------------------

    if len(max_points) > 0:
        clustering_max = DBSCAN(
            eps=0.02 * max(max_x - min_x, max_y - min_y),
            min_samples=1
        ).fit(max_points)

        labels_max = clustering_max.labels_

        unique_labels_max = np.unique(labels_max)
        maxima_x, maxima_y = [], []

        for lbl in unique_labels_max:
            cluster_pts = max_points[labels_max == lbl]
            maxima_x.append(cluster_pts[:, 0].mean())
            maxima_y.append(cluster_pts[:, 1].mean())

        maxima_z = [func(np.array([x, y])) for x, y in zip(maxima_x, maxima_y)]
    else:
        maxima_x, maxima_y, maxima_z = [], [], []

    # ---------------------------
    # Локальные максимумы (жёлтые точки)
    # ---------------------------

    maxima_trace = go.Scatter(
        x=maxima_x,
        y=maxima_y,
        mode='markers',
        customdata=maxima_z,
        hovertemplate=(
            "x: %{x:.4f}<br>"
            "y: %{y:.4f}<br>"
            "f(x,y): %{customdata:.6f}"
            "<extra></extra>"
        ),
        marker=dict(
            size=6,
            color='yellow',
            symbol='x'
        ),
        opacity=0.8,
        showlegend=False
    )

    # ---------------------------
    # Добавляем trace-ы первого поколения
    # ---------------------------
    for slot in range(max_groups):
        if slot < len(generations[0]):
            points = generations[0][slot]
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_vals = [func(p) for p in points]
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    customdata=z_vals,
                    hovertemplate=(
                        "x: %{x:.4f}<br>"
                        "y: %{y:.4f}<br>"
                        "f(x,y): %{customdata:.6f}"
                        "<extra></extra>"
                    ),
                    marker=dict(size=7, color=(
                        name_color if not has_color else
                        (color_pool[slot] if not predator_prey else
                         (predator_color if roles[0][slot] == "predator" else
                          prey_color)))),
                    name='',
                    showlegend=False
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[], y=[],
                    mode='markers',
                    marker=dict(size=7, color=(
                        name_color)),
                    name='',
                    showlegend=False,
                    visible=True
                )
            )

    fig.add_trace(minima_trace)
    fig.add_trace(maxima_trace)

    # ---------------------------
    # Создаём фреймы (только точки и аннотации)
    # ---------------------------
    frames = []
    for gen_idx, groups in enumerate(generations):
        frame_traces = []
        for slot in range(max_groups):
            if slot < len(groups):
                points = groups[slot]
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                z_vals = [func(p) for p in points]
                frame_traces.append(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    customdata=z_vals,
                    hovertemplate=(
                        "x: %{x:.4f}<br>"
                        "y: %{y:.4f}<br>"
                        "f(x,y): %{customdata:.6f}"
                        "<extra></extra>"
                    ),
                    marker=dict(size=7, color=(
                        name_color if not has_color else
                        (color_pool[slot] if not predator_prey else
                         (predator_color if roles[gen_idx][slot] == "predator" else
                          prey_color)))),
                    showlegend=False
                ))
            else:
                frame_traces.append(go.Scatter(
                    x=[], y=[],
                    mode='markers',
                    marker=dict(size=7, color=(
                        name_color)),
                    showlegend=False,
                    visible=True
                ))

        frame_traces.append(
            go.Scatter(
                x=minima_x,
                y=minima_y,
                mode='markers',
                customdata=minima_z,
                hovertemplate=(
                    "x: %{x:.4f}<br>"
                    "y: %{y:.4f}<br>"
                    "f(x,y): %{customdata:.6f}"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=6,
                    color='red',
                    symbol='x'
                ),
                opacity=0.5,
                showlegend=False
            )
        )

        frame_traces.append(
            go.Scatter(
                x=maxima_x,
                y=maxima_y,
                mode='markers',
                customdata=maxima_z,
                hovertemplate=(
                    "x: %{x:.4f}<br>"
                    "y: %{y:.4f}<br>"
                    "f(x,y): %{customdata:.6f}"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=6,
                    color='yellow',
                    symbol='x'
                ),
                opacity=0.8,
                showlegend=False
            )
        )

        annotation_text = (
                f"<b>Generation: {gen_idx + 1}</b><br>"
                f"Number of groups: {len(groups)}<br>" +
                "<br>".join([f"{i}) size {size}, mean: {ind:.3f}, min: {m:.3f}"
                             for i, (size, ind, m) in enumerate(group_names[gen_idx], start=1)])
        )

        frames.append(go.Frame(
            data=frame_traces,
            name=f"gen{gen_idx}",
            layout=go.Layout(
                annotations=[dict(
                    text=annotation_text,
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=1.05, y=0.95,
                    align="left",
                    xanchor="left",
                    yanchor="top"
                )]
            )
        ))

    fig.frames = frames

    # ---------------------------
    # Слайдер
    # ---------------------------
    steps = []
    for i, fr in enumerate(frames):
        steps.append(dict(
            method="animate",
            args=[[fr.name], {"mode": "immediate",
                              "frame": {"duration": 0, "redraw": True},
                              "transition": {"duration": 0}}],
            label=str(i)
        ))

    fig.update_layout(
        sliders=[dict(
            active=0,
            steps=steps,
            x=0.0, y=-0.00,
            len=1.0
        )],
        autosize=False,
        width=1000, height=800,
        title="Speciation in the genetic algorithm",
        xaxis_title="X",
        yaxis_title="Y",
        margin=dict(r=300),
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            constrain="domain"
        ),
    )

    fig.show(config={"responsive": False})


def reduce_generations(data, n_components=2):
    reducer = umap.UMAP(n_components=n_components)

    result = []

    for generation in data:
        flat_points = [p for group in generation for p in group]
        flat_points = np.array(flat_points)

        reduced = reducer.fit_transform(flat_points)

        gen_result = []
        idx = 0
        for group in generation:
            group_size = len(group)
            gen_result.append(reduced[idx: idx + group_size].tolist())
            idx += group_size

        result.append(gen_result)

    return result


from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SpeciesSnapshot:
    generation: int
    points: np.ndarray          # (Ni, D)
    mean_fitness: float
    min_fitness: float
    max_fitness: float
    role: str


@dataclass
class TrackedSpecies:
    id: int
    snapshots: List[SpeciesSnapshot] = field(default_factory=list)

    def last_points(self) -> np.ndarray:
        return self.snapshots[-1].points


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    #return np.linalg.norm(np.mean(a, axis=0) - np.mean(b, axis=0))
    # a: (N, D), b: (M, D)
    d_ab = np.mean(
        np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1)
    )
    d_ba = np.mean(
        np.min(np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2), axis=1)
    )
    return max(d_ab, d_ba)


def compute_distance_matrix(prev_sets, curr_sets):
    P = len(prev_sets)
    C = len(curr_sets)
    D = np.zeros((P, C))

    for i, a in enumerate(prev_sets):
        for j, b in enumerate(curr_sets):
            D[i, j] = chamfer_distance(a, b)

    return D


def mutual_nearest_neighbors(D):
    """
    Возвращает список пар (p, c),
    где p — индекс вида из t-1,
    c — индекс вида из t
    """
    prev_best = D.argmin(axis=1)   # для каждого prev -> лучший curr
    curr_best = D.argmin(axis=0)   # для каждого curr -> лучший prev

    matches = []
    for p, c in enumerate(prev_best):
        if curr_best[c] == p:
            matches.append((p, c))

    return matches


class SpeciesTracker:
    def __init__(self):
        self.tracked: Dict[int, TrackedSpecies] = {}
        self.active: Dict[int, TrackedSpecies] = {}
        self._next_id = 0

    def _process_generation(
            self,
            generation: int,
            populations: List[np.ndarray],
            fitnesses: List[np.ndarray],
            roles: List[str]
    ):
        curr_sets = populations

        prev_ids = list(self.active.keys())
        prev_sets = [self.active[tid].last_points() for tid in prev_ids]

        if prev_sets:
            D = compute_distance_matrix(prev_sets, curr_sets)
            matches = mutual_nearest_neighbors(D)
        else:
            matches = []

        assignment = {}

        for p_idx, c_idx in matches:
            assignment[c_idx] = prev_ids[p_idx]

        new_active = {}

        for j, (points, fit, role) in enumerate(
                zip(populations, fitnesses, roles)
        ):
            snapshot = SpeciesSnapshot(
                generation=generation,
                points=points,
                mean_fitness=fit.mean(),
                min_fitness=fit.min(),
                max_fitness=fit.max(),
                role=role
            )

            if j in assignment:
                tracked = self.tracked[assignment[j]]
            else:
                tracked = TrackedSpecies(id=self._next_id)
                self.tracked[self._next_id] = tracked
                self._next_id += 1

            tracked.snapshots.append(snapshot)
            new_active[tracked.id] = tracked

        self.active = new_active

    def build_from_logger(self, logger: SpeciesLogger):
        for gen, (pops, fits, roles) in enumerate(
            zip(logger.generations, logger.fitness, logger.roles)
        ):
            self._process_generation(
                generation=gen,
                populations=pops,
                fitnesses=fits,
                roles=roles
            )


def plot_species_fitness(tracker: SpeciesTracker):
    import matplotlib.pyplot as plt
    import math

    species = list(tracker.tracked.values())
    n = len(species)

    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4 * cols, 2.5 * rows),
        sharex=True
    )

    axes = np.atleast_1d(axes).flatten()

    for ax, sp in zip(axes, species):
        gens = [s.generation for s in sp.snapshots]
        mean = [s.mean_fitness for s in sp.snapshots]
        lo = [s.min_fitness for s in sp.snapshots]
        hi = [s.max_fitness for s in sp.snapshots]

        ax.plot(gens, mean, label="mean")
        ax.plot(gens, lo, "--", alpha=0.6)
        ax.plot(gens, hi, "--", alpha=0.6)

        ax.set_title(f"Species {sp.id}", fontsize=9)
        ax.grid(True)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def filter_species(tracker, min_lifetime: int = 8):
    """
    Оставляет только виды, живущие >= min_lifetime поколений.
    """
    long_lived = []
    short_lived = []

    for sp in tracker.tracked.values():
        if len(sp.snapshots) >= min_lifetime:
            long_lived.append(sp)
        else:
            short_lived.append(sp)

    return long_lived, short_lived


import matplotlib.pyplot as plt


def get_max_generation(species):
    return max(
        s.snapshots[-1].generation
        for s in species
    )


def aligned_series(snapshots, max_gen):
    """
    Возвращает массив длины max_gen + 1,
    заполненный значениями fitness или NaN.
    """
    mean = np.full(max_gen + 1, np.nan)
    lo = np.full(max_gen + 1, np.nan)
    hi = np.full(max_gen + 1, np.nan)

    for s in snapshots:
        g = s.generation
        mean[g] = s.mean_fitness
        lo[g] = s.min_fitness
        hi[g] = s.max_fitness

    return mean, lo, hi


def get_global_fitness_range(species):
    y_min = np.inf
    y_max = -np.inf

    for sp in species:
        for s in sp.snapshots:
            y_min = min(y_min, s.min_fitness)
            y_max = max(y_max, s.max_fitness)

    return y_min, y_max


class SpeciesPager:
    def __init__(self, species, per_page=6):
        self.species = species
        self.per_page = per_page
        self.page = 0
        self.max_page = (len(species) - 1) // per_page
        self.max_generation = get_max_generation(species)
        self.y_min, self.y_max = get_global_fitness_range(species)

        self.fig = None

    def show(self):
        self.fig = plt.figure(figsize=(10, 7))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        plt.show()

    def _draw(self):
        plt.clf()

        start = self.page * self.per_page
        end = min(start + self.per_page, len(self.species))
        subset = self.species[start:end]

        rows = len(subset)

        for i, sp in enumerate(subset):
            ax = plt.subplot(rows, 1, i + 1)

            gens = [s.generation for s in sp.snapshots]

            marker = "o" if len(gens) < 5 else None

            x = np.arange(self.max_generation + 1)
            mean, lo, hi = aligned_series(sp.snapshots, self.max_generation)

            ax.plot(x, mean, marker=marker, label="mean")
            ax.fill_between(x, lo, hi, alpha=0.3)
            ax.set_xlim(0, self.max_generation)
            pad = 0.05 * (self.y_max - self.y_min)
            ax.set_ylim(self.y_min - pad, self.y_max + pad)

            ax.set_title(
                f"Species {sp.id} | lifetime={len(sp.snapshots)}",
                fontsize=9
            )
            ax.grid(True)

        plt.suptitle(
            f"Species fitness | page {self.page + 1}/{self.max_page + 1}",
            fontsize=11
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.draw()

    def _on_key(self, event):
        if event.key == "right" and self.page < self.max_page:
            self.page += 1
            self._draw()
        elif event.key == "left" and self.page > 0:
            self.page -= 1
            self._draw()


def plot_short_lived_species(short_lived):
    if not short_lived:
        print("No short-lived species to plot.")
        return

    plt.figure(figsize=(8, 4))

    for sp in short_lived:
        gens = [s.generation for s in sp.snapshots]
        mean = [s.mean_fitness for s in sp.snapshots]
        plt.plot(gens, mean, alpha=0.25)

    plt.title(f"Short-lived species (count={len(short_lived)})")
    plt.xlabel("Generation")
    plt.ylabel("Mean fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def build_from_logger(species_logger: SpeciesLogger):
    number_of_generations = len(species_logger.generations)

    for i in range(1, number_of_generations):
        ...

def species_analysis(species_logger: SpeciesLogger):
    # 1. построение трекера
    tracker = SpeciesTracker()
    tracker.build_from_logger(species_logger)

    # 2. фильтрация
    long_lived, short_lived = filter_species(
        tracker,
        min_lifetime=5
    )

    # 3. pager для долгоживущих
    pager = SpeciesPager(long_lived, per_page=4)
    pager.show()

    # 4. отдельный график для короткоживущих
    # plot_short_lived_species(short_lived)


if __name__ == "__main__":
    print(px.colors.qualitative.Dark24)
