from typing import List
import math


def frechet_distance(a: List[float], b: List[float]) -> float:
    """Discrete Frechet distance implemented iteratively."""
    if not a or not b:
        return 0.0

    n, m = len(a), len(b)
    ca = [[0.0] * m for _ in range(n)]
    ca[0][0] = abs(a[0] - b[0])
    for i in range(1, n):
        ca[i][0] = max(ca[i - 1][0], abs(a[i] - b[0]))
    for j in range(1, m):
        ca[0][j] = max(ca[0][j - 1], abs(a[0] - b[j]))
    for i in range(1, n):
        for j in range(1, m):
            d = abs(a[i] - b[j])
            ca[i][j] = max(min(ca[i - 1][j], ca[i - 1][j - 1], ca[i][j - 1]), d)
    return ca[n - 1][m - 1]


def pearson_correlation(a: List[float], b: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    mean_a = sum(a[:n]) / n
    mean_b = sum(b[:n]) / n
    num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    den_a = math.sqrt(sum((a[i] - mean_a) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((b[i] - mean_b) ** 2 for i in range(n)))
    denom = den_a * den_b
    if denom == 0:
        return 0.0
    return num / denom
