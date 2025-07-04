from typing import List
import math

# lightweight replacements for numpy functionality
from . import simple_numpy as np


def _scalar_dist(x: float, y: float) -> float:
    """Absolute difference for scalar DTW distances."""
    return abs(x - y)


def dtw_distance(a: List[float], b: List[float]) -> float:
    """Compute classic DTW distance using dynamic programming."""
    n, m = len(a), len(b)
    cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i][j] = d + min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
    return cost[n][m]


def dtw_distance_with_radius(a: List[float], b: List[float], radius: int = 10) -> float:
    """DTW using a Sakoe-Chiba band with given radius."""
    n, m = len(a), len(b)
    radius = max(radius, abs(n - m))
    cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i][j] = d + min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])
    return cost[n][m]


def dtw_distance_normalized(a: List[float], b: List[float]) -> float:
    """Normalized DTW distance to compare sequences of different lengths."""
    d = dtw_distance(a, b)
    return d / (len(a) + len(b))


def dtw_with_window(a: List[float], b: List[float], window_size: int | None = None) -> float:
    """DTW with Sakoe-Chiba band constraint."""
    if window_size is None:
        min_len = min(len(a), len(b))
        window_size = max(5, int(0.05 * min_len))
    return dtw_distance_with_radius(a, b, window_size)


def dtw_distance_early_abandon(
    a: List[float],
    b: List[float],
    threshold: float = float("inf"),
) -> float:
    """Fixed early abandoning DTW."""
    n, m = len(a), len(b)
    cost = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    for i in range(1, n + 1):
        min_prev_row = min(cost[i - 1][j] for j in range(m + 1))
        if min_prev_row > threshold:
            return float("inf")  # Actually abandon!

        for j in range(1, m + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i][j] = d + min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])

    return cost[n][m]


def lb_keogh(s1: List[float], s2: List[float], r: int = 5) -> float:
    """LB_Keogh lower bound for DTW."""
    lb_sum = 0.0
    n2 = len(s2)
    if n2 == 0:
        return 0.0
    for i in range(len(s1)):
        start = max(0, i - r)
        end = min(n2, i + r + 1)
        if start >= end:
            lower = upper = s2[-1]
        else:
            window = s2[start:end]
            lower = min(window)
            upper = max(window)
        if s1[i] > upper:
            lb_sum += (s1[i] - upper) ** 2
        elif s1[i] < lower:
            lb_sum += (lower - s1[i]) ** 2
    return float(math.sqrt(lb_sum))
