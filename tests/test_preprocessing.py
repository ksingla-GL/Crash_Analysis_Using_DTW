import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import denoise_series


def test_denoise_series_length():
    series = [float(i % 5) for i in range(30)]
    noisy = [x + 0.1 for x in series]
    result = denoise_series(noisy, window_length=5, polyorder=2)
    assert len(result) == len(series)
