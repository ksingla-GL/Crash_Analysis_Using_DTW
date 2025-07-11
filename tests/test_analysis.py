import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.game_analysis import CrashGameAnalysis


def test_cluster_labels():
    analysis = CrashGameAnalysis('rounds_new1', max_files=4)
    labels = analysis.cluster(k=2, target_length=None, weights=(1.0, 0.5, 0.5, 0.0))
    assert len(labels) == 4


def test_analyze_clusters():
    analysis = CrashGameAnalysis('rounds_new1', max_files=4)
    labels = analysis.cluster(k=2, target_length=None, weights=(1.0, 0.5, 0.5, 0.0))
    stats = analysis.analyze_clusters(labels)
    assert set(stats.keys()) <= {0, 1}
    for info in stats.values():
        assert 'avg_max_multiplier' in info
        assert 'avg_length' in info


def test_evaluate_and_learn_weights():
    analysis = CrashGameAnalysis('rounds_new1', max_files=4)
    labels = analysis.cluster(k=2, target_length=None, weights=(1.0, 0.5, 0.5, 0.0))
    scores = analysis.evaluate_clusters(labels, (1.0, 0.5, 0.5, 0.0))
    assert 'silhouette' in scores and 'davies_bouldin' in scores
    weights = analysis.learn_weights(k=2, target_length=None)
    assert isinstance(weights, tuple) and len(weights) >= 3
