import random
from typing import List, Callable
import os

from . import simple_numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def _distance_task(
    i: int,
    j: int,
    seq_i: List[float],
    seq_j: List[float],
    dist_func: Callable[[List[float], List[float]], float],
) -> tuple[int, int, float]:
    """Compute distance between two sequences (helper for multiprocessing)."""
    return i, j, dist_func(seq_i, seq_j)


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


def _k_medoids_plus_plus_impl(
    data: List[List[float]],
    k: int,
    dist: Callable[[List[float], List[float]], float],
    max_iter: int,
) -> List[int]:
    """Internal implementation of k-medoids++ given a distance function."""
    n = len(data)
    medoid_indices = [random.randint(0, n - 1)]

    for _ in range(1, k):
        distances = []
        for i in range(n):
            if i in medoid_indices:
                continue
            min_dist = min(dist(data[i], data[m]) for m in medoid_indices)
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
            dists = [dist(series, data[m]) for m in medoid_indices]
            label = dists.index(min(dists))
            if labels[idx] != label:
                labels[idx] = label
                changed = True
        for i, m_idx in enumerate(list(medoid_indices)):
            cluster_points = [idx for idx, lbl in enumerate(labels) if lbl == i]
            if not cluster_points:
                continue
            best_idx = m_idx
            best_cost = sum(dist(data[p], data[m_idx]) for p in cluster_points)
            for p in cluster_points:
                cost = sum(dist(data[p], data[q]) for q in cluster_points)
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
    """K-medoids++ with distance caching"""
    n = len(data)
    if k <= 0 or k > n:
        raise ValueError("Invalid k")

    # Create distance cache
    dist_cache: dict[tuple[int, int], float] = {}

    def cached_dist(i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in dist_cache:
            dist_cache[key] = distance_func(data[i], data[j])
        return dist_cache[key]

    # k-medoids++ initialization
    medoid_indices = [random.randint(0, n - 1)]
    for _ in range(1, k):
        distances = []
        for i in range(n):
            if i in medoid_indices:
                continue
            min_dist = min(cached_dist(i, m) for m in medoid_indices)
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
        for idx in range(n):
            dists = [cached_dist(idx, m) for m in medoid_indices]
            label = dists.index(min(dists))
            if labels[idx] != label:
                labels[idx] = label
                changed = True
        for i, m_idx in enumerate(list(medoid_indices)):
            cluster_points = [idx for idx, lbl in enumerate(labels) if lbl == i]
            if not cluster_points:
                continue
            best_idx = m_idx
            best_cost = sum(cached_dist(p, m_idx) for p in cluster_points)
            for p in cluster_points:
                cost = sum(cached_dist(p, q) for q in cluster_points)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = p
            medoid_indices[i] = best_idx
        if not changed:
            break
    return labels


def parallel_distance_matrix(
    data: List[List[float]],
    distance_func: Callable[[List[float], List[float]], float],
    n_jobs: int = -1,
) -> List[List[float]]:
    """Compute pairwise distance matrix in parallel."""
    n = len(data)
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    dist_matrix = np.zeros((n, n))

    if n_jobs == 1:
        for i, j in pairs:
            dist = distance_func(data[i], data[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_distance_task, i, j, data[i], data[j], distance_func): (i, j)
                for i, j in pairs
            }
            for future in as_completed(futures):
                i, j = futures[future]
                dist = future.result()[2]
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist

    return dist_matrix


def k_medoids_with_matrix(
    n: int,
    k: int,
    matrix_distance: Callable[[int, int], float],
    max_iter: int = 30,
) -> List[int]:
    """K-medoids using a precomputed distance matrix."""
    medoids = random.sample(range(n), k)
    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            dists = [matrix_distance(i, m) for m in medoids]
            label = dists.index(min(dists))
            if labels[i] != label:
                labels[i] = label
                changed = True
        for idx, m in enumerate(list(medoids)):
            cluster_points = [i for i, l in enumerate(labels) if l == idx]
            if not cluster_points:
                continue
            best = m
            best_cost = sum(matrix_distance(p, m) for p in cluster_points)
            for p in cluster_points:
                cost = sum(matrix_distance(p, q) for q in cluster_points)
                if cost < best_cost:
                    best = p
                    best_cost = cost
            medoids[idx] = best
        if not changed:
            break
    return labels


def clara_clustering(
    data: List[List[float]],
    k: int,
    distance_func: Callable[[List[float], List[float]], float],
    sample_size: int = 40,
    n_samples: int = 5,
) -> List[int]:
    """CLARA algorithm for large scale k-medoids."""
    n = len(data)
    best_cost = float("inf")
    best_labels: List[int] | None = None

    for _ in range(n_samples):
        subset_indices = random.sample(range(n), min(sample_size, n))
        subset = [data[i] for i in subset_indices]
        sample_labels = k_medoids_plus_plus(subset, k, distance_func)

        medoids = []
        for c in range(k):
            cluster_idx = [i for i, lbl in enumerate(sample_labels) if lbl == c]
            if cluster_idx:
                medoids.append(subset_indices[cluster_idx[0]])

        labels = []
        total_cost = 0.0
        for i in range(n):
            dists = [distance_func(data[i], data[m]) for m in medoids]
            best_d = min(dists)
            labels.append(dists.index(best_d))
            total_cost += best_d

        if total_cost < best_cost:
            best_cost = total_cost
            best_labels = labels

    return best_labels if best_labels is not None else [0] * n
