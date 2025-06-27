import random
from typing import List, Callable


def k_medoids(
    data: List[List[float]],
    k: int,
    distance_func: Callable[[List[float], List[float]], float],
    max_iter: int = 10,
) -> List[int]:
    """Simplified k-medoids clustering."""
    if k <= 0 or k > len(data):
        raise ValueError("Invalid k")
    medoid_indices = random.sample(range(len(data)), k)
    labels = [0] * len(data)
    for _ in range(max_iter):
        changed = False
        # assign step
        for idx, series in enumerate(data):
            distances = [distance_func(series, data[m]) for m in medoid_indices]
            label = distances.index(min(distances))
            if labels[idx] != label:
                labels[idx] = label
                changed = True
        # update step
        for i, m_idx in enumerate(list(medoid_indices)):
            cluster_points = [idx for idx, lbl in enumerate(labels) if lbl == i]
            if not cluster_points:
                continue
            best_idx = m_idx
            best_cost = sum(distance_func(data[p], data[m_idx]) for p in cluster_points)
            for p in cluster_points:
                cost = sum(distance_func(data[p], data[q]) for q in cluster_points)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = p
            medoid_indices[i] = best_idx
        if not changed:
            break
    return labels


def k_medoids_plus_plus(
    data: List[List[float]],
    k: int,
    distance_func: Callable[[List[float], List[float]], float],
    max_iter: int = 30,
) -> List[int]:
    """K-medoids with k-means++ inspired initialization."""
    n = len(data)
    if k <= 0 or k > n:
        raise ValueError("Invalid k")

    medoid_indices = [random.randint(0, n - 1)]

    for _ in range(1, k):
        distances = []
        for i in range(n):
            if i in medoid_indices:
                continue
            min_dist = min(distance_func(data[i], data[m]) for m in medoid_indices)
            distances.append((i, min_dist))
        if not distances:
            break
        weights = [d[1] ** 2 for d in distances]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen = random.choices([d[0] for d in distances], weights=weights)[0]
        medoid_indices.append(chosen)

    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        for idx, series in enumerate(data):
            dists = [distance_func(series, data[m]) for m in medoid_indices]
            label = dists.index(min(dists))
            if labels[idx] != label:
                labels[idx] = label
                changed = True
        for i, m_idx in enumerate(list(medoid_indices)):
            cluster_points = [idx for idx, lbl in enumerate(labels) if lbl == i]
            if not cluster_points:
                continue
            best_idx = m_idx
            best_cost = sum(distance_func(data[p], data[m_idx]) for p in cluster_points)
            for p in cluster_points:
                cost = sum(distance_func(data[p], data[q]) for q in cluster_points)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = p
            medoid_indices[i] = best_idx
        if not changed:
            break
    return labels
