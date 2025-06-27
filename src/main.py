# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:07:53 2025

@author: kshit
"""

import os
from src.game_analysis import CrashGameAnalysis
from src.preprocessing import extract_features

# Analyze your sample data
analysis = CrashGameAnalysis('rounds_new1')

# Extract simple feature summary
all_features = []
for i, seq in enumerate(analysis.raw_data):
    features = extract_features(seq)
    features['file_id'] = analysis.file_ids[i]
    all_features.append(features)

print("=== Data Distribution ===")
print(f"Total rounds: {len(analysis.raw_data)}")
print(f"Length range: {min(len(s) for s in analysis.raw_data)} - {max(len(s) for s in analysis.raw_data)}")
instant_rate = sum(1 for s in analysis.raw_data if max(s) < 1.1) / len(analysis.raw_data) * 100
print(f"Instant crash rate: {instant_rate:.1f}%")

# Try different cluster counts
for k in [3, 4, 5]:
    print(f"\n=== {k} clusters analysis ===")

    labels = analysis.cluster(k=k, target_length=None, weights=(1.0, 0.0, 0.0))
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

    analysis.visualize_patterns(labels, output_dir=f"plots_k{k}")
    analysis.create_similarity_heatmap(labels, f"plots_k{k}/similarity_matrix.png")
