# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:07:53 2025

@author: kshit
"""

import os
import sys
from pathlib import Path

# Allow running this script directly from the project root by
# adding the parent directory to ``sys.path`` so that ``src`` can
# be imported without installing the package.  This also ensures
# the path is propagated to subprocesses when using multiprocessing.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.game_analysis import CrashGameAnalysis
from src.preprocessing import extract_features


def main() -> None:
    """Run a small analysis example on the sample data."""
    analysis = CrashGameAnalysis('clean_sample_rounds')

    # Extract simple feature summary
    all_features = []
    for i, seq in enumerate(analysis.raw_data):
        features = extract_features(seq)
        features['file_id'] = analysis.file_ids[i]
        all_features.append(features)

    print("=== Data Distribution ===")
    print(f"Total rounds: {len(analysis.raw_data)}")
    print(
        f"Length range: {min(len(s) for s in analysis.raw_data)} - {max(len(s) for s in analysis.raw_data)}"
    )
    instant_rate = (
        sum(1 for s in analysis.raw_data if max(s) < 1.1) / len(analysis.raw_data) * 100
    )
    print(f"Instant crash rate: {instant_rate:.1f}%")

    best_k, best_w, evals = analysis.determine_optimal_k([3, 4, 5])
    print("\n=== Cluster Evaluation ===")
    for k, scores in evals.items():
        print(
            f"k={k}: silhouette={scores['silhouette']:.3f} DB={scores['davies_bouldin']:.3f}"
        )

    print(f"\nOptimal k determined as {best_k} with weights {best_w}")
    labels = analysis.optimized_cluster(
        k=best_k,
        target_length=50,
        weights=best_w,
        use_clara=True,
        window_size=10,
    )
    stats = analysis.analyze_clusters(labels)
    pattern_names = analysis.identify_crash_patterns(labels)
    for cid, cluster_stats in stats.items():
        print(f"\nCluster {cid} - '{pattern_names.get(cid, f'Pattern_{cid}')}' :")
        print(f"  Avg max multiplier: {cluster_stats['avg_max_multiplier']:.2f}x")
        print(f"  Avg length: {cluster_stats['avg_length']:.0f} ticks")
        print(f"  Crash before 2x: {cluster_stats['crash_before_2x_rate']*100:.1f}%")
        print(f"  Examples: {', '.join(cluster_stats['examples'])}")

        exit_zones = analysis.find_exit_zones(cid, labels)
        if exit_zones:
            print(f"  Suggested exit zones: {[f'{z:.2f}x' for z in exit_zones]}")

    # Additional pattern analysis for task 5
    medoids = analysis.get_cluster_medoids(labels)
    barycenters = analysis.compute_barycenters(labels, target_length=50)
    continuation = analysis.analyze_continuation_tendencies(labels)
    distribution = analysis.pattern_distribution_by_strata(labels)

    print("\n=== Cluster Medoids and Barycenters ===")
    for cid in sorted(medoids):
        print(
            f"Cluster {cid}: medoid len {len(medoids[cid])}, barycenter len {len(barycenters[cid])}"
        )

    print("\n=== Pattern Continuation Tendencies ===")
    for cid, slope in continuation.items():
        print(f"Cluster {cid}: avg post-peak slope {slope:.4f}")

    print("\n=== Pattern Distribution by Length Strata ===")
    for stratum, counts in distribution.items():
        summary = ', '.join(f"{cid}: {cnt}" for cid, cnt in counts.items())
        print(f"{stratum}: {summary}")

    analysis.visualize_patterns(labels, output_dir=f"plots_k{best_k}")
    analysis.create_similarity_heatmap(labels, f"plots_k{best_k}/similarity_matrix.png")

    # Compare patterns across multiple resampling scales
    scales_result = analysis.compare_resampling_scales(
        best_k, [50, 100, 200], best_w
    )
    if "similarity_matrix" in scales_result:
        print("\nCross-scale similarity matrix computed")

    # Stratified analysis and plotting
    strata_stats = analysis.analyze_by_length_strata(k=best_k)
    for stratum, clusters in strata_stats.items():
        print(f"Stratum '{stratum}' contains {len(clusters)} clusters")
    analysis.plot_stratified_patterns(output_dir=f"plots_k{best_k}")


if __name__ == "__main__":
    main()
