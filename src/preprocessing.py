from typing import List
import math


def paa(series: List[float], segments: int) -> List[float]:
    """Piecewise Aggregate Approximation."""
    if segments <= 0:
        raise ValueError("segments must be positive")
    n = len(series)
    if n == 0:
        return []
    step = n / segments
    result = []
    for i in range(segments):
        start = int(round(i * step))
        end = int(round((i + 1) * step))
        if end > n:
            end = n
        if start >= end:
            result.append(series[start if start < n else n - 1])
        else:
            segment = series[start:end]
            result.append(sum(segment) / len(segment))
    return result


def zscore(series: List[float]) -> List[float]:
    n = len(series)
    if n == 0:
        return []
    mean = sum(series) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in series) / n)
    if std == 0:
        return [0.0 for _ in series]
    return [(x - mean) / std for x in series]


def resample_series(series: List[float], target_length: int) -> List[float]:
    """Resample series to target length using PAA."""
    return paa(series, target_length)

import numpy as np
from typing import Dict


def count_plateaus(series: List[float], threshold: float = 0.01) -> int:
    """Count stable regions in the series."""
    if len(series) < 3:
        return 0
    plateaus = 0
    for i in range(1, len(series) - 1):
        if abs(series[i] - series[i - 1]) < threshold and abs(series[i + 1] - series[i]) < threshold:
            plateaus += 1
    return plateaus


def count_momentum_changes(series: List[float]) -> int:
    """Count momentum direction changes."""
    if len(series) < 3:
        return 0
    changes = 0
    last_sign = 0
    for i in range(1, len(series)):
        diff = series[i] - series[i - 1]
        sign = 1 if diff > 0 else -1 if diff < 0 else 0
        if last_sign != 0 and sign != 0 and sign != last_sign:
            changes += 1
        if sign != 0:
            last_sign = sign
    return changes


def extract_features(series: List[float]) -> Dict[str, float]:
    """Extract crash-relevant features from a series."""
    if not series:
        return {
            'max_multiplier': 0.0,
            'time_to_peak': 0.0,
            'volatility': 0.0,
            'crash_velocity': 0.0,
            'plateau_count': 0.0,
            'momentum_changes': 0.0,
        }
    max_val = max(series)
    max_idx = series.index(max_val)
    volatility = float(np.std(np.diff(series))) if len(series) > 1 else 0.0
    crash_velocity = series[-1] - series[-2] if len(series) > 1 else 0.0
    return {
        'max_multiplier': max_val,
        'time_to_peak': float(max_idx),
        'volatility': float(volatility),
        'crash_velocity': float(crash_velocity),
        'plateau_count': float(count_plateaus(series)),
        'momentum_changes': float(count_momentum_changes(series)),
    }


def adaptive_resample(series: List[float], min_length: int = 50, max_length: int = 200) -> List[float]:
    """Adaptive resampling depending on sequence length."""
    n = len(series)
    if n < min_length and n > 0:
        x = np.linspace(0, n - 1, min_length)
        return np.interp(x, range(n), series).tolist()
    if n > max_length:
        return paa(series, max_length)
    return series
