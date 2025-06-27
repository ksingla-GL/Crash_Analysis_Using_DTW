from typing import List


def kelly_fraction(win_prob: float, win_return: float, loss_return: float) -> float:
    """Compute Kelly fraction with crash risk adjustment."""
    if loss_return == 0:
        return 0.0
    b = win_return / abs(loss_return)
    fraction = win_prob - (1 - win_prob) / b
    return max(0.0, fraction)


def scale_position(base: float, risk_fraction: float) -> float:
    return base * risk_fraction
