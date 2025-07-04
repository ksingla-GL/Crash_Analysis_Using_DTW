from typing import List, Callable, Sequence
from collections import defaultdict


def _pairwise_distance(data: List[Sequence[float]], dist_func: Callable[[Sequence[float], Sequence[float]], float]):
    cache = {}
    def d(i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        if key not in cache:
            cache[key] = dist_func(data[i], data[j])
        return cache[key]
    return d


def silhouette_score(data: List[Sequence[float]], labels: List[int], dist_func: Callable[[Sequence[float], Sequence[float]], float]) -> float:
    n = len(data)
    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)
    if n <= 1 or len(clusters) == 1:
        return 0.0
    d = _pairwise_distance(data, dist_func)
    scores = []
    for i in range(n):
        lbl = labels[i]
        same = [j for j in clusters[lbl] if j != i]
        if same:
            a = sum(d(i, j) for j in same) / len(same)
        else:
            a = 0.0
        b_vals = []
        for other_lbl, idxs in clusters.items():
            if other_lbl == lbl:
                continue
            b = sum(d(i, j) for j in idxs) / len(idxs)
            b_vals.append(b)
        b = min(b_vals) if b_vals else 0.0
        score = 0.0 if a == b == 0 else (b - a) / max(a, b)
        scores.append(score)
    return sum(scores) / len(scores)


def davies_bouldin_index(data: List[Sequence[float]], labels: List[int], dist_func: Callable[[Sequence[float], Sequence[float]], float]) -> float:
    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)
    n_clusters = len(clusters)
    if n_clusters <= 1:
        return 0.0
    d = _pairwise_distance(data, dist_func)
    medoids = {}
    scat = {}
    for lbl, idxs in clusters.items():
        best_i = idxs[0]
        best_cost = float('inf')
        for i in idxs:
            cost = sum(d(i, j) for j in idxs)
            if cost < best_cost:
                best_cost = cost
                best_i = i
        medoids[lbl] = best_i
        scat[lbl] = sum(d(i, best_i) for i in idxs) / len(idxs) if idxs else 0.0
    labels_list = list(clusters.keys())
    db = 0.0
    for i in labels_list:
        max_ratio = 0.0
        for j in labels_list:
            if i == j:
                continue
            denom = d(medoids[i], medoids[j])
            ratio = (scat[i] + scat[j]) / denom if denom > 0 else float('inf')
            if ratio > max_ratio:
                max_ratio = ratio
        db += max_ratio
    return db / n_clusters
