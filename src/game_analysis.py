import csv
import os
from typing import List, Tuple, Callable, Optional, Dict

from .preprocessing import (
    resample_series,
    zscore,
    adaptive_resample,
    extract_features,
)
from .metrics import frechet_distance, pearson_correlation
from .dtw import dtw_distance, dtw_with_window
from .clustering import k_medoids_plus_plus
from .position import kelly_fraction, scale_position


class CrashGameAnalysis:
    def __init__(self, file_path: str, max_files: Optional[int] = None):
        self.raw_data, self.file_ids = self._load_data(file_path, max_files)
        self.preprocessed_data: Optional[List[List[float]]] = None

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
        """Apply adaptive resampling followed by optional fixed-length resampling and z-score normalization."""
        sequences = [adaptive_resample(seq) for seq in self.raw_data]
        if target_length is not None:
            sequences = [resample_series(seq, target_length) for seq in sequences]
        return [zscore(seq) for seq in sequences]

    def distance(self,
        a: List[float],
        b: List[float],
        weights: Tuple[float, float, float],
        window_size: Optional[int] = None,
    ) -> float:
        """Weighted distance between two sequences."""
        w_dtw, w_frechet, w_corr = weights
        if window_size is None:
            d_dtw = dtw_distance(a, b)
        else:
            d_dtw = dtw_with_window(a, b, window_size)
        d_frechet = frechet_distance(a, b)
        corr = pearson_correlation(a, b)
        d_corr = 1 - corr
        return w_dtw * d_dtw + w_frechet * d_frechet + w_corr * d_corr

    def cluster(self,
        k: int,
        target_length: Optional[int],
        weights: Tuple[float, float, float],
        window_size: Optional[int] = None,
    ) -> List[int]:
        """Cluster preprocessed sequences using k-medoids++ initialization."""
        preprocessed = self.preprocess(target_length)
        self.preprocessed_data = preprocessed
        dist_func: Callable[[List[float], List[float]], float] = (
            lambda x, y: self.distance(x, y, weights, window_size)
        )
        labels = k_medoids_plus_plus(preprocessed, k, dist_func)
        return labels

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
    
    def _find_medoid(self, indices: List[int]) -> int:
        """Find medoid (most central point) for given indices."""
        data = self.preprocessed_data if self.preprocessed_data is not None else self.preprocess(None)
        best_idx = indices[0]
        best_cost = float("inf")

        for i in indices:
            cost = sum(
                dtw_distance(data[i], data[j]) for j in indices if i != j
            )
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        
        return best_idx

    def visualize_patterns(self, labels: List[int], output_dir: str = "plots") -> None:
        """Visualize clusters with time-normalized x-axis."""
        import matplotlib.pyplot as plt
        import numpy as np
        
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
        import numpy as np

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
        import numpy as np

        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        crash_multipliers: List[float] = []
        for idx in indices:
            seq = self.raw_data[idx]
            for i in range(1, len(seq)):
                if seq[i] < seq[i - 1] * 0.9:
                    crash_multipliers.append(seq[i - 1])
                    break

        if crash_multipliers:
            return np.percentile(crash_multipliers, [25, 50, 75]).tolist()
        return []

    def create_similarity_heatmap(self, labels: List[int], output_path: str) -> None:
        """Create heatmap of DTW distances between cluster medoids."""
        import numpy as np
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
                    dist_matrix[i, j] = dtw_distance(
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
