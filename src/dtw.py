from typing import List
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtw_distance(a: List[float], b: List[float]) -> float:
    """Compute DTW distance using FastDTW algorithm."""
    distance, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))
    return distance


def dtw_distance_with_radius(a: List[float], b: List[float], radius: int = 10) -> float:
    """FastDTW with custom radius for speed/accuracy tradeoff."""
    distance, _ = fastdtw(a, b, radius=radius, dist=euclidean)
    return distance


def dtw_distance_normalized(a: List[float], b: List[float]) -> float:
    """Normalized DTW distance to compare sequences of different lengths."""
    distance, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))
    return distance / (len(a) + len(b))


def dtw_with_window(a: List[float], b: List[float], window_size: int | None = None) -> float:
    """DTW with a Sakoe-Chiba band constraint."""
    if window_size is None:
        window_size = max(10, int(0.1 * max(len(a), len(b))))
    distance, _ = fastdtw(a, b, radius=window_size, dist=euclidean)
    return distance
