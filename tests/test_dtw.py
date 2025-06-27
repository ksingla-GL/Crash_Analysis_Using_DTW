import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dtw import dtw_distance


def test_dtw_distance_zero():
    assert dtw_distance([1, 2, 3], [1, 2, 3]) == 0


def test_dtw_distance_symmetry():
    a = [1, 2, 3]
    b = [2, 3, 4]
    assert dtw_distance(a, b) == dtw_distance(b, a)
