import csv
import os
from typing import List, Tuple, Callable, Optional, Dict

from .preprocessing import (
    resample_series,
    zscore_vectorized,
    adaptive_resample,
    extract_features,
    extract_features_batch,
)

from . import simple_numpy as np
from .metrics import frechet_distance, pearson_correlation
from .dtw import (
    dtw_distance,
    dtw_with_window,
    dtw_distance_early_abandon,
    lb_keogh,
)
from .clustering import (
    k_medoids_plus_plus,
    parallel_distance_matrix,
    clara_clustering,
    k_medoids_with_matrix,
)
from .position import kelly_fraction, scale_position
from .validation import silhouette_score, davies_bouldin_index
from concurrent.futures import ThreadPoolExecutor


class DistanceWrapper:
    """Picklable wrapper for weighted distance computation."""

    def __init__(self, weights: Tuple[float, float, float], window_size: Optional[int] = None):
        self.weights = weights
        self.window_size = window_size

    def __call__(self, a: List[float], b: List[float]) -> float:
        w_dtw, w_frechet, w_corr = self.weights[:3]
        if self.window_size is None:
            w_size = max(5, int(0.05 * min(len(a), len(b))))
        else:
            w_size = self.window_size

        lb = lb_keogh(a, b)
        d_dtw = dtw_distance_early_abandon(a, b, threshold=lb * 2)
        if d_dtw == float("inf"):
            d_dtw = dtw_with_window(a, b, w_size)

        d_frechet = frechet_distance(a, b)
        corr = pearson_correlation(a, b)
        d_corr = 1 - corr

        return w_dtw * d_dtw + w_frechet * d_frechet + w_corr * d_corr


class CrashGameAnalysis:
    def __init__(
        self,
        file_path: str,
        max_files: Optional[int] = None,
        sg_window_length: int = 5,
        sg_polyorder: int = 2,
    ):
        """Load raw data and set preprocessing parameters."""
        self.raw_data, self.file_ids = self._load_data(file_path, max_files)
        self.sg_window_length = sg_window_length
        self.sg_polyorder = sg_polyorder
        self.preprocessed_data: Optional[List[List[float]]] = None
        self.feature_data: Optional[List[Dict[str, float]]] = extract_features_batch(self.raw_data)
        self._distance_cache: Dict[str, float] = {}

    @staticmethod
    def _load_data(
        path: str, max_files: Optional[int] = None
    ) -> Tuple[List[List[float]], List[str]]:
        data: List[List[float]] = []
        ids: List[str] = []
        if os.path.isdir(path):
            csv_files: List[str] = []
            for root, _, files in os.walk(path):
                for name in files:
                    if name.endswith(".csv"):
                        csv_files.append(os.path.join(root, name))
            csv_files.sort()
            if max_files is not None:
                csv_files = csv_files[:max_files]
            for fp in csv_files:
                with open(fp, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and "multiplier" in reader.fieldnames:
                        seq = [float(row["multiplier"]) for row in reader]
                    else:
                        seq = [float(x) for row in reader for x in row]
                    data.append(seq)
                    ids.append(os.path.splitext(os.path.basename(fp))[0])
        else:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        seq = [float(x) for x in row]
                    except ValueError:
                        continue
                    data.append(seq)
                    ids.append(str(len(ids)))
        return data, ids

    def preprocess(self, target_length: Optional[int] = None) -> List[List[float]]:
        """Optimized adaptive resampling"""
        # Skip denoising for sequences < 5 ticks
        sequences = [seq if len(seq) < 5 else adaptive_resample(seq) for seq in self.raw_data]

        if target_length is not None:
            sequences = [resample_series(seq, target_length) for seq in sequences]
        return [zscore_vectorized(seq) for seq in sequences]

    def _get_cache_key(self, idx1: int, idx2: int) -> str:
        """Create a consistent key for caching distances."""
        return f"{min(idx1, idx2)}_{max(idx1, idx2)}"

    def _cached_distance(
        self,
        idx1: int,
        idx2: int,
        data: List[List[float]],
        weights: Tuple[float, float, float],
        window_size: Optional[int] = None,
    ) -> float:
        key = self._get_cache_key(idx1, idx2)
        if key not in self._distance_cache:
            self._distance_cache[key] = self.distance(
                data[idx1], data[idx2], weights, window_size
            )
        return self._distance_cache[key]

    def distance(self,
        a: List[float],
        b: List[float],
        weights: Tuple[float, float, float],
        window_size: Optional[int] = None,
    ) -> float:
        """Weighted distance between two sequences."""
        w_dtw, w_frechet, w_corr = weights[:3]
        if window_size is None:
            window_size = max(5, int(0.05 * min(len(a), len(b))))
        lb = lb_keogh(a, b)
        d_dtw = dtw_distance_early_abandon(a, b, threshold=lb * 2)
        if d_dtw == float("inf"):
            d_dtw = dtw_with_window(a, b, window_size)
        d_frechet = frechet_distance(a, b)
        corr = pearson_correlation(a, b)
        d_corr = 1 - corr
        dist = w_dtw * d_dtw + w_frechet * d_frechet + w_corr * d_corr
        if len(weights) > 3:
            import math

            if not hasattr(self, "feature_cache"):
                self.feature_cache = {}
                if self.preprocessed_data is not None:
                    for seq in self.preprocessed_data:
                        self.feature_cache[id(seq)] = list(
                            extract_features(seq).values()
                        )

            def get_features(seq: List[float]) -> List[float]:
                fid = id(seq)
                if fid not in self.feature_cache:
                    self.feature_cache[fid] = list(extract_features(seq).values())
                return self.feature_cache[fid]

            w_feat = weights[3]
            fa = get_features(a)
            fb = get_features(b)
            d_feat = math.sqrt(sum((fa[i] - fb[i]) ** 2 for i in range(len(fa))))
            dist += w_feat * d_feat
        return dist

    def cluster(self,
        k: int,
        target_length: Optional[int],
        weights: Tuple[float, ...],
        window_size: Optional[int] = None,
    ) -> List[int]:
        """Cluster preprocessed sequences using k-medoids++ initialization."""
        preprocessed = self.preprocess(target_length)
        self.preprocessed_data = preprocessed
        dist_func: Callable[[List[float], List[float]], float] = (
            lambda x, y: self.distance(x, y, weights, window_size)
        )
        # k_medoids_plus_plus already caches distances internally
        labels = k_medoids_plus_plus(preprocessed, k, dist_func)
        return labels

    def optimized_cluster(
        self,
        k: int,
        target_length: Optional[int],
        weights: Tuple[float, float, float],
        use_clara: bool = True,
        n_jobs: int = -1,
        window_size: Optional[int] = None,
    ) -> List[int]:
        """Cluster using cached distances and optional parallelization."""

        self._distance_cache.clear()
        preprocessed = self.preprocess(target_length)
        self.preprocessed_data = preprocessed

        if len(preprocessed) > 500 and use_clara:
            dist_func = DistanceWrapper(weights, window_size)
            return clara_clustering(preprocessed, k, dist_func)

        dist_func = DistanceWrapper(weights, window_size)
        dist_matrix = parallel_distance_matrix(preprocessed, dist_func, n_jobs)

        def matrix_distance(i: int, j: int) -> float:
            return dist_matrix[i][j]

        return k_medoids_with_matrix(len(preprocessed), k, matrix_distance)

    def optimal_fraction(self, win_prob: float, win_return: float, loss_return: float, base: float) -> float:
        f = kelly_fraction(win_prob, win_return, loss_return)
        return scale_position(base, f)

    def analyze_clusters(self, labels: List[int]) -> Dict[int, Dict[str, float]]:
        """Generate simple statistics for each cluster."""
        stats: Dict[int, Dict[str, float]] = {}
        for cid in sorted(set(labels)):
            indices = [i for i, lbl in enumerate(labels) if lbl == cid]
            seqs = [self.raw_data[i] for i in indices]
            if not seqs:
                continue
            max_mult = [max(seq) if seq else 0.0 for seq in seqs]
            avg_max = sum(max_mult) / len(max_mult)
            lengths = [len(seq) for seq in seqs]
            avg_len = sum(lengths) / len(lengths)
            crash_rate = sum(1 for m in max_mult if m < 2.0) / len(max_mult)
            examples = [self.file_ids[i] for i in indices[:3]]
            stats[cid] = {
                "avg_max_multiplier": avg_max,
                "avg_length": avg_len,
                "crash_before_2x_rate": crash_rate,
                "examples": examples,
            }
        return stats

    def evaluate_clusters(
        self,
        labels: List[int],
        weights: Tuple[float, ...],
        window_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute silhouette and Davies-Bouldin indices."""
        data = (
            self.preprocessed_data
            if self.preprocessed_data is not None
            else self.preprocess(None)
        )
        dist_func: Callable[[List[float], List[float]], float] = (
            lambda x, y: self.distance(x, y, weights, window_size)
        )
        sil = silhouette_score(data, labels, dist_func)
        db = davies_bouldin_index(data, labels, dist_func)
        return {"silhouette": sil, "davies_bouldin": db}

    def learn_weights(
        self,
        k: int,
        target_length: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> Tuple[float, ...]:
        """Return fixed weights for a fast ensemble distance."""

        # In the accelerated mode we skip grid search and just
        # use the ensemble weighting of all three distances.
        return (0.5, 0.3, 0.2)
    
    def _find_medoid(self, indices: List[int]) -> int:
        """Parallel medoid finding"""
        from concurrent.futures import ThreadPoolExecutor

        data = self.preprocessed_data

        with ThreadPoolExecutor() as executor:
            futures = {}
            for i in indices:
                futures[
                    executor.submit(
                        lambda i: sum(
                            dtw_distance(data[i], data[j]) for j in indices if i != j
                        ),
                        i,
                    )
                ] = i

            best_idx = indices[0]
            best_cost = float("inf")
            for future in futures:
                cost = future.result()
                if cost < best_cost:
                    best_cost = cost
                    best_idx = futures[future]
        return best_idx


    def visualize_patterns(self, labels: List[int], output_dir: str = "plots") -> None:
        """Visualize clusters with time-normalized x-axis."""
        import matplotlib.pyplot as plt
        from . import simple_numpy as np
        
        os.makedirs(output_dir, exist_ok=True)
        
        data = (
            self.preprocessed_data
            if self.preprocessed_data is not None
            else self.preprocess(None)
        )

        for cid in sorted(set(labels)):
            indices = [i for i, lbl in enumerate(labels) if lbl == cid]
            if not indices:
                continue
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Top plot: Time-normalized (0 to 1) using preprocessed data
            for idx in indices:
                seq = data[idx]
                x = np.linspace(0, 1, len(seq))  # Normalize time to [0,1]
                ax1.plot(x, seq, color="gray", alpha=0.3)
            
            # Plot medoid
            medoid_idx = self._find_medoid(indices)
            medoid_seq = data[medoid_idx]
            x = np.linspace(0, 1, len(medoid_seq))
            ax1.plot(
                x,
                medoid_seq,
                color="red",
                linewidth=2,
                label=f"Medoid: {self.file_ids[medoid_idx]}",
            )

            # Add volatility bands around the medoid
            std_dev: List[float] = []
            for j in range(len(medoid_seq)):
                vals = [data[i][j] for i in indices if j < len(data[i])]
                std_dev.append(np.std(vals))
            upper_band = [m + s for m, s in zip(medoid_seq, std_dev)]
            lower_band = [m - s for m, s in zip(medoid_seq, std_dev)]
            ax1.fill_between(x, lower_band, upper_band, color="red", alpha=0.2)

            # Suggested exit zones
            exit_zones = self.find_exit_zones(cid, labels)
            for exit_mult in exit_zones:
                ax1.axhline(y=exit_mult, color="red", linestyle="--", alpha=0.5)
            
            ax1.set_title(f"Cluster {cid} - Time Normalized")
            ax1.set_xlabel("Normalized Time (0-1)")
            ax1.set_ylabel("Multiplier")
            ax1.legend()
            
            # Bottom plot: Actual time scale (first 100 ticks)
            for idx in indices:
                seq = self.raw_data[idx][:100]  # First 100 ticks only
                ax2.plot(seq, color="gray", alpha=0.3)
            
            ax2.set_title(f"Cluster {cid} - First 100 Ticks")
            ax2.set_xlabel("Tick Number")
            ax2.set_ylabel("Multiplier")

            for exit_mult in exit_zones:
                ax2.axhline(y=exit_mult, color="red", linestyle="--", alpha=0.5)
            
            # Add statistics
            stats = self.analyze_clusters(labels)[cid]
            textstr = f"Avg Max: {stats['avg_max_multiplier']:.2f}x\n"
            textstr += f"Avg Length: {stats['avg_length']:.0f} ticks\n"
            textstr += f"Crash <2x: {stats['crash_before_2x_rate']*100:.0f}%"
            ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"cluster_{cid}.png"))
            plt.close()

    def identify_crash_patterns(self, labels: List[int]) -> Dict[int, str]:
        """Identify crash pattern types for each cluster."""
        from . import simple_numpy as np

        pattern_names: Dict[int, str] = {}
        for cid in set(labels):
            indices = [i for i, lbl in enumerate(labels) if lbl == cid]
            if not indices:
                continue
            seqs = [self.raw_data[i] for i in indices]
            avg_max = np.mean([max(seq) if seq else 0.0 for seq in seqs])
            avg_len = np.mean([len(seq) for seq in seqs])
            early_crash_rate = sum(1 for seq in seqs if len(seq) < 50) / len(seqs)

            if avg_max < 1.3 and early_crash_rate > 0.7:
                pattern_names[cid] = "Quick Crash"
            elif avg_max > 10:
                pattern_names[cid] = "Moon Shot"
            elif 2 <= avg_max <= 5 and avg_len > 200:
                pattern_names[cid] = "Steady Climber"
            elif avg_max > 3 and avg_len < 100:
                pattern_names[cid] = "Volatile Spike"
            else:
                pattern_names[cid] = f"Pattern_{cid}"

        return pattern_names

    def find_exit_zones(self, cluster_id: int, labels: List[int]) -> List[float]:
        """Identify suggested exit multipliers for a cluster."""
        from . import simple_numpy as np

        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        crash_multipliers: List[float] = []
        for idx in indices:
            seq = self.raw_data[idx]
            for i in range(1, len(seq)):
                if seq[i] < seq[i - 1] * 0.9:
                    crash_multipliers.append(seq[i - 1])
                    break

        if crash_multipliers:
            # simple_numpy.percentile already returns a Python list
            return np.percentile(crash_multipliers, [25, 50, 75])
        return []

    def create_similarity_heatmap(self, labels: List[int], output_path: str) -> None:
        """Create heatmap of DTW distances between cluster medoids."""
        from . import simple_numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = (
            self.preprocessed_data
            if self.preprocessed_data is not None
            else self.preprocess(None)
        )

        unique_labels = sorted(set(labels))
        medoid_indices = [
            self._find_medoid([i for i, lbl in enumerate(labels) if lbl == cid])
            for cid in unique_labels
        ]

        n = len(medoid_indices)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix[i][j] = dtw_distance(
                        data[medoid_indices[i]],
                        data[medoid_indices[j]],
                    )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            dist_matrix,
            annot=True,
            cmap="coolwarm",
            xticklabels=[f"C{c}" for c in unique_labels],
            yticklabels=[f"C{c}" for c in unique_labels],
        )
        plt.title("Pattern Similarity Matrix (DTW Distance)")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def compare_resampling_scales(
        self, k: int, scales: List[int], weights: Tuple[float, ...]
    ) -> Dict[int, List[List[float]]]:
        """Cluster patterns at different resampling scales and return medoids."""
        from . import simple_numpy as np

        results: Dict[int, List[List[float]]] = {}
        for scale in scales:
            labels = self.cluster(k, target_length=scale, weights=weights)
            medoids = [
                self.preprocessed_data[self._find_medoid([i for i, lbl in enumerate(labels) if lbl == cid])]
                for cid in sorted(set(labels))
            ]
            results[scale] = medoids
        # Compute cross scale similarity matrix
        if len(scales) > 1:
            n = sum(len(m) for m in results.values())
            matrix = np.zeros((n, n))
            all_medoids: List[List[float]] = []
            for scale in scales:
                all_medoids.extend(results[scale])
            for i in range(n):
                for j in range(n):
                    if i != j:
                        matrix[i][j] = dtw_distance(all_medoids[i], all_medoids[j])
            # 'matrix' is already a nested list from simple_numpy.zeros
            results["similarity_matrix"] = matrix
        return results

    def determine_optimal_k(
        self, k_values: List[int], target_length: Optional[int] = None, window_size: Optional[int] = None
    ) -> Tuple[int, Tuple[float, ...], Dict[int, Dict[str, float]]]:
        """FAST k selection using fixed weights but full evaluation metrics."""

        eval_results: Dict[int, Dict[str, float]] = {}
        best_k = k_values[0]
        best_db = float("inf")

        # Predefined ensemble weights (DTW, Frechet, Correlation)
        weights = self.learn_weights(k_values[0], target_length, window_size)

        for k in k_values:
            labels = self.optimized_cluster(
                k,
                target_length,
                weights,
                use_clara=True,
                window_size=window_size,
            )
            scores = self.evaluate_clusters(labels, weights, window_size)
            eval_results[k] = scores
            if scores["davies_bouldin"] < best_db:
                best_db = scores["davies_bouldin"]
                best_k = k

        return best_k, weights, eval_results

    def stratify_by_length(
        self, bins: Optional[List[int]] = None
    ) -> Dict[str, List[int]]:
        """Group sequence indices into length-based strata."""
        if bins is None:
            bins = [50, 200, 500, float("inf")]

        names = ["short", "medium", "long", "extra_long"]
        strata: Dict[str, List[int]] = {n: [] for n in names}
        for idx, seq in enumerate(self.raw_data):
            l = len(seq)
            if l < bins[0]:
                strata[names[0]].append(idx)
            elif l < bins[1]:
                strata[names[1]].append(idx)
            elif l < bins[2]:
                strata[names[2]].append(idx)
            else:
                strata[names[3]].append(idx)

        self.strata_indices = strata
        self.strata = {name: [self.raw_data[i] for i in inds] for name, inds in strata.items()}
        return strata

    def _preprocess_sequences(
        self, sequences: List[List[float]], target_length: Optional[int] = None
    ) -> List[List[float]]:
        """Preprocess a list of sequences."""
        seqs = [seq if len(seq) < 5 else adaptive_resample(seq) for seq in sequences]
        if target_length is not None:
            seqs = [resample_series(s, target_length) for s in seqs]
        return [zscore_vectorized(s) for s in seqs]

    def analyze_by_length_strata(
        self, k: int = 3, weights: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Cluster and analyze patterns within length-based strata."""
        if weights is None:
            weights = (0.5, 0.3, 0.2)

        strata_results: Dict[str, Dict[int, Dict[str, float]]] = {}
        strata = self.stratify_by_length()

        for stratum, indices in strata.items():
            if not indices:
                continue
            subset = [self.raw_data[i] for i in indices]
            processed = self._preprocess_sequences(subset, target_length=None)
            dist_func = lambda x, y: self.distance(x, y, weights)
            k_use = min(k, len(processed))
            labels = k_medoids_plus_plus(processed, k_use, dist_func)

            stats: Dict[int, Dict[str, float]] = {}
            for cid in sorted(set(labels)):
                sub_inds = [i for i, lbl in enumerate(labels) if lbl == cid]
                seqs = [subset[i] for i in sub_inds]
                if not seqs:
                    continue
                max_mult = [max(seq) if seq else 0.0 for seq in seqs]
                avg_max = sum(max_mult) / len(max_mult)
                lengths = [len(seq) for seq in seqs]
                avg_len = sum(lengths) / len(lengths)
                crash_rate = sum(1 for m in max_mult if m < 2.0) / len(max_mult)
                examples = [self.file_ids[indices[i]] for i in sub_inds[:3]]
                stats[cid] = {
                    "avg_max_multiplier": avg_max,
                    "avg_length": avg_len,
                    "crash_before_2x_rate": crash_rate,
                    "examples": examples,
                }

            strata_results[stratum] = stats

        return strata_results

    def plot_stratified_patterns(self, output_dir: str = "plots") -> None:
        """Visualize patterns grouped by length strata."""
        import matplotlib.pyplot as plt
        from . import simple_numpy as np

        if not hasattr(self, "strata"):
            self.stratify_by_length()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        for ax, (stratum, data) in zip(axes.flat, self.strata.items()):
            for seq in data:
                x = np.linspace(0, 1, len(seq))
                ax.plot(x, seq, color="gray", alpha=0.3)
            ax.set_title(f"{stratum} patterns")
            ax.set_xlabel("Normalized Time")
            ax.set_ylabel("Multiplier")

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "stratified_patterns.png"))
        plt.close()
