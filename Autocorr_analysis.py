import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# ============================================
# RNG PREDICTION ANALYSIS
# ============================================

def load_rounds(base_dir='clean_sample_rounds'):
    """Load all rounds."""
    rounds = {}
    for folder in sorted(Path(base_dir).iterdir()):
        if folder.is_dir():
            csv_files = list(folder.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                mult_col = [c for c in df.columns if 'multiplier' in c.lower()][0]
                rounds[folder.name] = df[mult_col].values
    return rounds

def extract_rng_parameters(rounds):
    """Try to reverse-engineer RNG parameters from game data."""
    
    # Store extracted parameters
    all_drifts = []
    big_moves = []
    round_params = []
    
    for round_id, data in rounds.items():
        if len(data) < 10:
            continue
            
        # Calculate tick-to-tick changes
        changes = np.diff(data) / data[:-1]
        
        # Separate big moves from normal drift
        normal_changes = []
        big_changes = []
        
        for change in changes:
            if abs(change) > 0.1:  # Big move threshold
                big_changes.append(change)
            else:
                normal_changes.append(change)
        
        if normal_changes:
            # Estimate drift parameters for this round
            drift_mean = np.mean(normal_changes)
            drift_std = np.std(normal_changes)
            
            round_params.append({
                'round': round_id,
                'drift_mean': drift_mean,
                'drift_std': drift_std,
                'n_big_moves': len(big_changes),
                'big_move_rate': len(big_changes) / len(changes) if changes.any() else 0,
                'length': len(data),
                'max_mult': np.max(data)
            })
            
            all_drifts.extend(normal_changes)
            big_moves.extend(big_changes)
    
    return pd.DataFrame(round_params), all_drifts, big_moves

def analyze_rng_patterns(round_params):
    """Look for patterns in extracted RNG parameters."""
    
    print("="*60)
    print("RNG PARAMETER ANALYSIS")
    print("="*60)
    
    # 1. Drift Distribution Analysis
    print("\n1. DRIFT DISTRIBUTION:")
    print(f"Mean drift: {round_params['drift_mean'].mean():.4f}")
    print(f"Std of drift means: {round_params['drift_mean'].std():.4f}")
    
    # Expected drift if uniform(-0.02, 0.03)
    expected_mean = (-0.02 + 0.03) / 2
    print(f"Expected mean (if uniform): {expected_mean:.4f}")
    
    # Test if drift means follow a pattern
    drift_means = round_params['drift_mean'].values
    
    # Autocorrelation test - do consecutive rounds have similar drift?
    if len(drift_means) > 100:
        autocorr = np.corrcoef(drift_means[:-1], drift_means[1:])[0,1]
        print(f"\nDrift autocorrelation (consecutive rounds): {autocorr:.4f}")
        
        if abs(autocorr) > 0.1:
            print("WARNING: Drift may be predictable between rounds!")
    
    # 2. Big Move Rate Analysis
    print("\n2. BIG MOVE FREQUENCY:")
    mean_big_rate = round_params['big_move_rate'].mean()
    print(f"Average big move rate: {mean_big_rate:.4f}")
    print(f"Expected rate: 0.1250")
    
    # Test if big move rate varies predictably
    big_rate_std = round_params['big_move_rate'].std()
    print(f"Std of big move rates: {big_rate_std:.4f}")
    
    return drift_means

def test_seed_predictability(rounds):
    """Test if we can predict seeds/biases from sequences."""
    
    print("\n3. SEED PREDICTABILITY TEST:")
    print("If seeds are predictable, early rounds should predict later ones...")
    
    # Group rounds into batches (simulating sequential play)
    sorted_rounds = sorted(rounds.items())
    batch_size = 100
    
    predictions = []
    
    for i in range(0, len(sorted_rounds) - batch_size, batch_size):
        # Use batch to predict next batch
        train_batch = sorted_rounds[i:i+batch_size]
        test_batch = sorted_rounds[i+batch_size:i+2*batch_size]
        
        if len(test_batch) < batch_size:
            continue
        
        # Extract features from train batch
        train_drifts = []
        for _, data in train_batch:
            if len(data) > 10:
                changes = np.diff(data[:10]) / data[:9]  # First 10 ticks
                train_drifts.append(np.mean(changes))
        
        # Extract same from test batch
        test_drifts = []
        for _, data in test_batch:
            if len(data) > 10:
                changes = np.diff(data[:10]) / data[:9]
                test_drifts.append(np.mean(changes))
        
        if train_drifts and test_drifts:
            # Predict test batch mean from train batch mean
            train_mean = np.mean(train_drifts)
            test_mean = np.mean(test_drifts)
            
            predictions.append({
                'batch': i // batch_size,
                'train_mean': train_mean,
                'test_mean': test_mean,
                'error': abs(train_mean - test_mean)
            })
    
    pred_df = pd.DataFrame(predictions)
    
    if len(pred_df) > 0:
        # Check if predictions are better than random
        correlation = pred_df['train_mean'].corr(pred_df['test_mean'])
        print(f"\nCorrelation between consecutive batch means: {correlation:.4f}")
        
        if abs(correlation) > 0.3:
            print("STRONG PATTERN DETECTED! Seeds may be predictable!")
        elif abs(correlation) > 0.1:
            print("Weak pattern detected. Further investigation needed.")
        else:
            print("No predictable pattern in seeds.")
    
    return pred_df

def analyze_crash_patterns(rounds):
    """Analyze if crash probability follows expected distribution."""
    
    print("\n4. CRASH PROBABILITY ANALYSIS:")
    
    # Expected crash rate per tick: 0.5%
    # Probability of surviving N ticks: (0.995)^N
    
    survival_data = []
    
    for length in [10, 20, 50, 100, 200]:
        rounds_surviving = sum(1 for data in rounds.values() if len(data) >= length)
        total_rounds = len(rounds)
        observed_survival = rounds_surviving / total_rounds
        expected_survival = 0.995 ** length
        
        survival_data.append({
            'ticks': length,
            'observed': observed_survival,
            'expected': expected_survival,
            'ratio': observed_survival / expected_survival
        })
    
    survival_df = pd.DataFrame(survival_data)
    print("\nSurvival rates:")
    print(survival_df)
    
    # Chi-squared test - Fixed version
    # Scale expected to match observed total
    observed_counts = survival_df['observed'].values * len(rounds)
    expected_counts = survival_df['expected'].values * len(rounds)
    
    # Ensure sums match (fix rounding issues)
    expected_counts = expected_counts * (observed_counts.sum() / expected_counts.sum())
    
    chi2, p_value = stats.chisquare(observed_counts, expected_counts)
    
    print(f"\nChi-squared test: χ² = {chi2:.2f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("Crash distribution differs from expected!")
    else:
        print("Crash distribution matches expected 0.5% per tick")
    
    return survival_df

def plot_rng_analysis(round_params, drift_means, pred_df):
    """Visualize RNG analysis results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Drift distribution
    ax1.hist(round_params['drift_mean'], bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.005, color='r', linestyle='--', label='Expected mean')
    ax1.set_xlabel('Average Drift per Round')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Drift Means')
    ax1.legend()
    
    # 2. Drift autocorrelation
    if len(drift_means) > 100:
        ax2.scatter(drift_means[:-1][:500], drift_means[1:][:500], alpha=0.5, s=10)
        ax2.set_xlabel('Drift Mean (round n)')
        ax2.set_ylabel('Drift Mean (round n+1)')
        ax2.set_title('Consecutive Round Drift Correlation')
        
        # Add correlation line
        z = np.polyfit(drift_means[:-1], drift_means[1:], 1)
        p = np.poly1d(z)
        x_line = np.linspace(drift_means.min(), drift_means.max(), 100)
        ax2.plot(x_line, p(x_line), "r--", alpha=0.8)
    
    # 3. Big move rate distribution
    ax3.hist(round_params['big_move_rate'], bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0.125, color='r', linestyle='--', label='Expected (12.5%)')
    ax3.set_xlabel('Big Move Rate')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Big Move Frequency Distribution')
    ax3.legend()
    
    # 4. Batch prediction accuracy
    if len(pred_df) > 0:
        ax4.scatter(pred_df['train_mean'], pred_df['test_mean'], alpha=0.7)
        ax4.plot([-0.01, 0.02], [-0.01, 0.02], 'r--', label='Perfect prediction')
        ax4.set_xlabel('Train Batch Mean Drift')
        ax4.set_ylabel('Test Batch Mean Drift')
        ax4.set_title('Seed Predictability Test')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('rng_analysis.png', dpi=150)
    plt.show()

def run_complete_rng_analysis():
    """Run complete RNG prediction analysis."""
    
    print("Loading crash game data...")
    rounds = load_rounds()
    print(f"Loaded {len(rounds)} rounds")
    
    # Extract RNG parameters
    round_params, all_drifts, big_moves = extract_rng_parameters(rounds)
    
    # Analyze patterns
    drift_means = analyze_rng_patterns(round_params)
    
    # Test seed predictability
    pred_df = test_seed_predictability(rounds)
    
    # Analyze crash patterns
    survival_df = analyze_crash_patterns(rounds)
    
    # Visualize
    plot_rng_analysis(round_params, drift_means, pred_df)
    
    # Final verdict
    print("\n" + "="*60)
    print("FINAL RNG ANALYSIS VERDICT:")
    print("="*60)
    
    # Check all indicators
    exploitable = False
    
    if len(drift_means) > 100:
        autocorr = np.corrcoef(drift_means[:-1], drift_means[1:])[0,1]
        if abs(autocorr) > 0.1:
            print("Drift shows autocorrelation - possibly exploitable!")
            exploitable = True
    
    if len(pred_df) > 0:
        correlation = pred_df['train_mean'].corr(pred_df['test_mean'])
        if abs(correlation) > 0.2:
            print("Batch patterns detected - seeds may be predictable!")
            exploitable = True
    
    mean_drift = round_params['drift_mean'].mean()
    if abs(mean_drift - 0.005) > 0.002:
        print(f" Drift bias detected: {mean_drift:.4f} vs expected 0.005")
        exploitable = True
    
    if not exploitable:
        print("No exploitable RNG patterns found")
        print("The game appears to use proper randomization")
    else:
        print("\n RECOMMENDATION: Further investigate detected patterns!")
        print("Consider tracking sequences of rounds for predictive signals")

# ============================================
# ADVANCED ANALYSIS FUNCTIONS
# ============================================

def analyze_seed_sequences(rounds, sequence_length=5):
    """Analyze if consecutive game seeds show patterns."""
    
    print("\n5. SEED SEQUENCE ANALYSIS:")
    print(f"Analyzing sequences of {sequence_length} consecutive rounds...")
    
    sorted_rounds = sorted(rounds.items())
    sequence_features = []
    
    for i in range(len(sorted_rounds) - sequence_length):
        sequence = sorted_rounds[i:i+sequence_length]
        
        # Extract features from sequence
        features = {
            'seq_start': i,
            'avg_length': np.mean([len(data) for _, data in sequence]),
            'avg_max_mult': np.mean([np.max(data) for _, data in sequence]),
            'crash_count': sum(1 for _, data in sequence if np.max(data) < 2.0),
            'big_win_count': sum(1 for _, data in sequence if np.max(data) > 10.0)
        }
        
        sequence_features.append(features)
    
    seq_df = pd.DataFrame(sequence_features)
    
    # Look for patterns in sequences
    if len(seq_df) > 100:
        # Check if crash streaks are more common than expected
        expected_crash_streak = 0.41 ** sequence_length  # 41% crash before 2x
        observed_all_crash = (seq_df['crash_count'] == sequence_length).mean()
        
        print(f"\nAll-crash sequences:")
        print(f"Expected: {expected_crash_streak:.4f}")
        print(f"Observed: {observed_all_crash:.4f}")
        
        if observed_all_crash > expected_crash_streak * 1.5:
            print("Crash streaks more common than expected!")
        
        # Check for hot/cold streaks
        hot_streaks = (seq_df['big_win_count'] >= 2).mean()
        print(f"\nHot streaks (2+ big wins): {hot_streaks:.4f}")
    
    return seq_df

def analyze_timing_patterns(rounds):
    """Check if game timing affects outcomes."""
    
    print("\n6. TIMING PATTERN ANALYSIS:")
    
    # Extract game IDs which contain timestamps
    timing_data = []
    
    for round_id, data in rounds.items():
        # Extract hour from game ID if format is like "20250625-69270bffb6994c17"
        try:
            if '-' in round_id:
                date_part = round_id.split('-')[0]
                if len(date_part) == 8:  # YYYYMMDD format
                    # We don't have hour info in this format, skip timing analysis
                    pass
        except:
            pass
    
    print("Note: Timing analysis requires timestamp data in game IDs")
    
    return None

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    run_complete_rng_analysis()
    
    # Run additional analyses
    rounds = load_rounds()
    
    # Analyze seed sequences
    seq_df = analyze_seed_sequences(rounds)
    
    # Analyze timing patterns
    timing_df = analyze_timing_patterns(rounds)
