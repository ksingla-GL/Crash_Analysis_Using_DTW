import math
from typing import Iterable, List, Sequence


def array(seq: Sequence[float]) -> List[float]:
    return list(seq)


def mean(seq: Sequence[float]) -> float:
    return sum(seq) / len(seq) if len(seq) else 0.0


def std(seq: Sequence[float]) -> float:
    if not seq:
        return 0.0
    m = mean(seq)
    return math.sqrt(sum((x - m) ** 2 for x in seq) / len(seq))


def zeros(shape: Sequence[int]) -> List[List[float]] | List[float]:
    if isinstance(shape, int):
        return [0.0 for _ in range(shape)]
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    if len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError("Unsupported shape")


def full(shape: Sequence[int], value: float) -> List[List[float]] | List[float]:
    if isinstance(shape, int):
        return [value for _ in range(shape)]
    if len(shape) == 1:
        return [value for _ in range(shape[0])]
    if len(shape) == 2:
        return [[value for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError("Unsupported shape")


def sqrt(x: float) -> float:
    return math.sqrt(x)


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [float(start)] * max(1, num)
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def interp(x: Iterable[float], xp: Sequence[float], fp: Sequence[float]) -> List[float]:
    xp = list(xp)
    fp = list(fp)
    if len(xp) != len(fp):
        raise ValueError("xp and fp must have same length")
    result = []
    for xv in x:
        if xv <= xp[0]:
            result.append(fp[0])
            continue
        if xv >= xp[-1]:
            result.append(fp[-1])
            continue
        for i in range(len(xp) - 1):
            if xp[i] <= xv <= xp[i + 1]:
                fraction = (xv - xp[i]) / (xp[i + 1] - xp[i])
                result.append(fp[i] * (1 - fraction) + fp[i + 1] * fraction)
                break
    return result


def percentile(data: Sequence[float], q: Sequence[float]) -> List[float]:
    if not data:
        return [0.0 for _ in q]
    d = sorted(data)
    n = len(d)
    result = []
    for qi in q:
        pos = (n - 1) * qi / 100.0
        lower = d[int(math.floor(pos))]
        upper = d[int(math.ceil(pos))]
        if lower == upper:
            result.append(lower)
        else:
            frac = pos - math.floor(pos)
            result.append(lower + (upper - lower) * frac)
    return result


def diff(seq: Sequence[float]) -> List[float]:
    return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]


def abs(arr: Iterable[float]) -> List[float]:
    return [abs(x) for x in arr]

