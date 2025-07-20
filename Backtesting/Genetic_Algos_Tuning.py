#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 10:57:14 2025

@author: Kshitij
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

@dataclass
class GeneticParams:
    """Parameters to optimize via genetic algorithm"""
    # Entry thresholds
    momentum_entry: float = 2.0  # Default: 2.0
    post_massive_min: float = 1.05  # Default: 1.05
    post_massive_max: float = 1.5  # Default: 1.5
    hot_streak_4_min: float = 1.0  # Default: 1.0
    hot_streak_4_max: float = 2.0  # Default: 2.0
    
    # Targets
    momentum_target_mult: float = 10.0  # Default: 20.0 / 2.0 = 10x
    post_massive_target_mult: float = 1.5  # Default: 1.5x
    hot_streak_4_target: float = 3.0  # Default: 3.0
    hot_streak_5_target: float = 5.0  # Default: 5.0
    
    # Stop losses
    momentum_stop: float = 0.8  # Default: 0.8
    post_massive_stop: float = 0.9  # Default: 0.9
    hot_streak_stop: float = 0.85  # Default: 0.85
    
    # Base bet sizes
    momentum_base_bet: float = 0.008  # Default: 0.008
    post_massive_base_bet: float = 0.015  # Default: 0.015
    hot_streak_base_bet: float = 0.010  # Default: 0.010
    hot_streak_5_base_bet: float = 0.014  # Default: 0.014
    
    # Other parameters
    massive_ago_limit: int = 3  # Default: 3
    hot_streak_count: int = 4  # Default: 4
    
    def mutate(self, mutation_rate=0.1, conservative=True):
        """Mutate parameters with constraints to avoid extreme values"""
        mutated = GeneticParams()
        
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if random.random() < mutation_rate:
                if conservative:
                    # Smaller mutations to stay near reasonable values
                    if 'entry' in field or 'min' in field or 'max' in field:
                        value *= random.uniform(0.9, 1.1)  # ±10% instead of ±20%
                    elif 'target' in field:
                        value *= random.uniform(0.85, 1.15)  # ±15% instead of ±30%
                    elif 'stop' in field:
                        value *= random.uniform(0.95, 1.05)  # ±5% instead of ±10%
                    elif 'bet' in field:
                        value *= random.uniform(0.85, 1.15)  # ±15% instead of ±30%
                    elif field == 'massive_ago_limit':
                        value = max(2, min(5, value + random.randint(-1, 1)))
                    elif field == 'hot_streak_count':
                        value = random.choice([3, 4, 5])
                else:
                    # Original mutation ranges
                    if 'entry' in field or 'min' in field or 'max' in field:
                        value *= random.uniform(0.8, 1.2)
                    elif 'target' in field:
                        value *= random.uniform(0.7, 1.3)
                    elif 'stop' in field:
                        value *= random.uniform(0.9, 1.1)
                    elif 'bet' in field:
                        value *= random.uniform(0.7, 1.3)
                    elif field == 'massive_ago_limit':
                        value = max(1, value + random.randint(-1, 1))
                    elif field == 'hot_streak_count':
                        value = random.randint(3, 5)
            
            # Apply constraints to keep parameters reasonable
            if 'stop' in field:
                value = max(0.5, min(0.95, value))  # Stops between 50-95%
            elif 'bet' in field:
                value = max(0.002, min(0.03, value))  # Bets between 0.2-3%
            elif 'momentum_entry' in field:
                value = max(1.5, min(3.0, value))  # Momentum entry 1.5-3.0x
            elif 'target_mult' in field:
                value = max(1.2, min(20.0, value))  # Targets 1.2-20x
                
            setattr(mutated, field, value)
        
        return mutated
    
    def crossover(self, other):
        """Uniform crossover with another parameter set"""
        child = GeneticParams()
        
        for field in self.__dataclass_fields__:
            # 50/50 chance to take from each parent
            if random.random() < 0.5:
                setattr(child, field, getattr(self, field))
            else:
                setattr(child, field, getattr(other, field))
        
        return child
    
    def regularization_penalty(self):
        """Penalty for extreme parameters (to avoid overfitting)"""
        penalty = 0.0
        defaults = GeneticParams()
        
        # Penalize large deviations from defaults
        for field in self.__dataclass_fields__:
            default_val = getattr(defaults, field)
            current_val = getattr(self, field)
            
            if isinstance(default_val, (int, float)):
                # Relative deviation
                deviation = abs(current_val - default_val) / (default_val + 0.001)
                penalty += deviation * 0.01  # Small penalty
        
        return penalty

class OptimizedBot:
    """Bot that uses genetic parameters"""
    
    def __init__(self, params: GeneticParams, capital=1000.0):
        self.params = params
        self.capital = capital
        self.init_capital = capital
        self.trades = []
        self.peak_capital = capital
        
        # Tracking
        self.last_max = None
        self.massive_ago = 999
        self.recent_pnls = deque(maxlen=10)
        self.recent_round_maxes = deque(maxlen=5)
        
        # Strategy tracking
        self.momentum_pnls = deque(maxlen=20)
        self.post_massive_pnls = deque(maxlen=30)
        self.hot_streak_pnls = deque(maxlen=20)
        
        self.n_trades = 0
        self.n_wins = 0
        self.entry_tick = 20
    
    def should_trade(self, entry_price):
        """Use genetic parameters for entry decisions"""
        # Momentum
        if entry_price >= self.params.momentum_entry:
            target = entry_price * self.params.momentum_target_mult
            return 'momentum', target, self.params.momentum_stop
        
        # Skip crashed
        if entry_price < 1.0:
            return None, 0, 0
        
        # Post-massive
        if (self.massive_ago <= self.params.massive_ago_limit and 
            self.params.post_massive_min <= entry_price < self.params.post_massive_max):
            target = entry_price * self.params.post_massive_target_mult
            return 'post_massive', target, self.params.post_massive_stop
        
        # Hot streaks
        if len(self.recent_round_maxes) >= self.params.hot_streak_count:
            streak_count = sum(1 for max_val in self.recent_round_maxes if max_val >= 2.0)
            
            if streak_count >= 5 and self.params.hot_streak_4_min <= entry_price < self.params.hot_streak_4_max:
                return 'hot_streak_5', self.params.hot_streak_5_target, self.params.hot_streak_stop
            elif streak_count >= self.params.hot_streak_count and self.params.hot_streak_4_min <= entry_price < self.params.hot_streak_4_max:
                return 'hot_streak', self.params.hot_streak_4_target, self.params.hot_streak_stop
        
        return None, 0, 0
    
    def get_bet_size(self, strat_type):
        """Use genetic parameters for bet sizing"""
        # Base sizes from params
        if strat_type == 'momentum':
            base = self.params.momentum_base_bet
            recent_pnls = self.momentum_pnls
        elif strat_type == 'post_massive':
            base = self.params.post_massive_base_bet
            recent_pnls = self.post_massive_pnls
        elif strat_type == 'hot_streak':
            base = self.params.hot_streak_base_bet
            recent_pnls = self.hot_streak_pnls
        else:  # hot_streak_5
            base = self.params.hot_streak_5_base_bet
            recent_pnls = self.hot_streak_pnls
        
        mult = 1.0
        
        # Base multipliers
        if strat_type == 'momentum':
            mult = 1.5
        elif strat_type == 'hot_streak_5':
            mult = 1.2
        
        # Performance adjustments (simplified for speed)
        if len(recent_pnls) >= 10:
            win_rate = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
            if win_rate > 0.3:
                mult *= 1.3
            elif win_rate < 0.1:
                mult *= 0.7
        
        # Drawdown protection
        dd = (self.peak_capital - self.capital) / self.peak_capital
        if dd > 0.05:
            mult *= max(1.0 - dd, 0.5)
        
        bet = self.capital * base * mult
        return min(bet, self.capital * 0.05) if bet >= 1.0 else 0

def fast_backtest(rounds_dict, params: GeneticParams) -> Dict:
    """Fast backtest for genetic algorithm"""
    bot = OptimizedBot(params)
    
    # Convert dict to list for consistent ordering
    if isinstance(rounds_dict, dict):
        sorted_rounds = sorted(rounds_dict.items())
    else:
        sorted_rounds = rounds_dict
    
    for i, (round_id, data) in enumerate(sorted_rounds[:-1]):
        # Track round stats
        if data is not None and len(data) > 0:
            round_max = max(data)
            bot.recent_round_maxes.append(round_max)
            
            if round_max >= 10.0:
                bot.massive_ago = 0
            else:
                bot.massive_ago += 1
        
        # Next round
        next_data = sorted_rounds[i + 1][1]
        if next_data is None or len(next_data) <= bot.entry_tick:
            continue
        
        # Entry
        entry_price = next_data[bot.entry_tick]
        strat_type, target, stop = bot.should_trade(entry_price)
        
        if strat_type is None:
            continue
        
        # Bet
        bet = bot.get_bet_size(strat_type)
        if bet <= 0:
            continue
        
        bot.n_trades += 1
        bot.capital -= bet
        
        # Find exit (simplified)
        exit_price = 0
        for tick in range(bot.entry_tick + 1, len(next_data)):
            curr_price = next_data[tick]
            ratio = curr_price / entry_price
            
            # Check exits
            if curr_price >= target or ratio <= stop or ratio > 10:
                exit_price = curr_price
                break
            elif tick - bot.entry_tick > 200:  # Simplified timeout
                exit_price = curr_price
                break
        
        # P&L
        pnl = bet * (exit_price / entry_price) - bet
        bot.capital += bet + pnl
        
        # Track
        bot.recent_pnls.append(pnl)
        if strat_type == 'momentum':
            bot.momentum_pnls.append(pnl)
        elif strat_type == 'post_massive':
            bot.post_massive_pnls.append(pnl)
        else:
            bot.hot_streak_pnls.append(pnl)
        
        if bot.capital > bot.peak_capital:
            bot.peak_capital = bot.capital
        
        if pnl > 0:
            bot.n_wins += 1
    
    # Calculate fitness metrics
    total_return = (bot.capital - bot.init_capital) / bot.init_capital
    max_dd = (bot.peak_capital - bot.capital) / bot.peak_capital if bot.peak_capital > 0 else 0
    win_rate = bot.n_wins / bot.n_trades if bot.n_trades > 0 else 0
    
    # Sharpe-like ratio (simplified)
    if len(bot.recent_pnls) > 0:
        returns = [p / bot.init_capital for p in bot.recent_pnls]
        sharpe = np.mean(returns) / (np.std(returns) + 0.0001) if returns else 0
    else:
        sharpe = 0
    
    return {
        'return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'n_trades': bot.n_trades,
        'sharpe': sharpe,
        'final_capital': bot.capital,
        'fitness': total_return * (1 - max_dd) * (1 + sharpe)  # Combined fitness
    }

class RobustGeneticOptimizer:
    """Genetic algorithm with proper train/validation/test split"""
    
    def __init__(self, all_rounds, train_ratio=0.6, val_ratio=0.2, 
                 population_size=50, elite_size=10):
        # Split data
        rounds_list = sorted(all_rounds.items())
        n_rounds = len(rounds_list)
        
        train_end = int(n_rounds * train_ratio)
        val_end = int(n_rounds * (train_ratio + val_ratio))
        
        self.train_rounds = rounds_list[:train_end]
        self.val_rounds = rounds_list[train_end:val_end]
        self.test_rounds = rounds_list[val_end:]
        
        print(f"Data split: Train={len(self.train_rounds)}, Val={len(self.val_rounds)}, Test={len(self.test_rounds)}")
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.generation = 0
        self.best_params = None
        self.best_val_fitness = -float('inf')
        self.history = []
        self.early_stop_patience = 10
        self.generations_without_improvement = 0
        
    def create_initial_population(self) -> List[GeneticParams]:
        """Create initial population with default + variations"""
        population = []
        
        # Always include default params
        population.append(GeneticParams())
        
        # Add some hand-tuned variations
        slightly_aggressive = GeneticParams()
        slightly_aggressive.momentum_target_mult = 15.0
        slightly_aggressive.hot_streak_5_target = 7.0
        population.append(slightly_aggressive)
        
        conservative = GeneticParams()
        conservative.momentum_base_bet = 0.006
        conservative.post_massive_base_bet = 0.012
        population.append(conservative)
        
        # Fill rest with random variations
        while len(population) < self.population_size:
            params = GeneticParams()
            # Start from default and mutate
            for _ in range(2):  # Less aggressive initial mutations
                params = params.mutate(mutation_rate=0.3, conservative=True)
            population.append(params)
        
        return population
    
    def evaluate_population(self, population: List[GeneticParams]) -> List[Tuple[GeneticParams, Dict, Dict]]:
        """Evaluate on training AND validation sets"""
        results = []
        
        for i, params in enumerate(population):
            if i % 10 == 0:
                print(f"  Evaluating individual {i+1}/{len(population)}...")
            
            # Train performance
            train_metrics = fast_backtest(self.train_rounds, params)
            
            # Validation performance  
            val_metrics = fast_backtest(self.val_rounds, params)
            
            # Apply regularization penalty
            reg_penalty = params.regularization_penalty()
            train_metrics['fitness'] -= reg_penalty
            val_metrics['fitness'] -= reg_penalty
            
            results.append((params, train_metrics, val_metrics))
        
        # Sort by VALIDATION fitness (not training!)
        results.sort(key=lambda x: x[2]['fitness'], reverse=True)
        return results
    
    def create_next_generation(self, evaluated_pop: List[Tuple[GeneticParams, Dict, Dict]]) -> List[GeneticParams]:
        """Create next generation with elitism and breeding"""
        next_gen = []
        
        # Keep elite based on validation performance
        for i in range(self.elite_size):
            next_gen.append(evaluated_pop[i][0])
        
        # Always keep the default params for stability
        next_gen.append(GeneticParams())
        
        # Fill rest with crossover and mutation
        while len(next_gen) < self.population_size:
            # Tournament selection based on validation fitness
            parent1 = self.tournament_select(evaluated_pop)
            parent2 = self.tournament_select(evaluated_pop)
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation (more conservative as we progress)
            mutation_rate = 0.2 * (1 - self.generation / 50)  # Decay mutation rate
            if random.random() < mutation_rate:
                child = child.mutate(conservative=True)
            
            next_gen.append(child)
        
        return next_gen
    
    def tournament_select(self, evaluated_pop: List[Tuple[GeneticParams, Dict, Dict]], 
                         tournament_size=3) -> GeneticParams:
        """Tournament selection based on validation fitness"""
        # Only select from top half to maintain quality
        top_half = evaluated_pop[:len(evaluated_pop)//2]
        tournament = random.sample(top_half, min(tournament_size, len(top_half)))
        winner = max(tournament, key=lambda x: x[2]['fitness'])  # Use validation fitness
        return winner[0]
    
    def optimize(self, generations=50):
        """Run genetic algorithm with early stopping"""
        print(f"Starting robust genetic optimization...")
        print(f"Population: {self.population_size}, Elite: {self.elite_size}")
        
        # Initial population
        population = self.create_initial_population()
        
        for gen in range(generations):
            print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate
            evaluated = self.evaluate_population(population)
            
            # Track best
            best_individual = evaluated[0]
            best_train = best_individual[1]
            best_val = best_individual[2]
            
            # Check for improvement
            if best_val['fitness'] > self.best_val_fitness:
                self.best_val_fitness = best_val['fitness']
                self.best_params = best_individual[0]
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
            
            # Log progress
            avg_train_fitness = np.mean([x[1]['fitness'] for x in evaluated])
            avg_val_fitness = np.mean([x[2]['fitness'] for x in evaluated])
            
            print(f"  Best train: Return={best_train['return']*100:.1f}%, DD={best_train['max_dd']*100:.1f}%")
            print(f"  Best val: Return={best_val['return']*100:.1f}%, DD={best_val['max_dd']*100:.1f}%")
            print(f"  Avg train fitness: {avg_train_fitness:.4f}")
            print(f"  Avg val fitness: {avg_val_fitness:.4f}")
            print(f"  Generations without improvement: {self.generations_without_improvement}")
            
            self.history.append({
                'generation': gen + 1,
                'best_train_fitness': best_train['fitness'],
                'best_val_fitness': best_val['fitness'],
                'avg_train_fitness': avg_train_fitness,
                'avg_val_fitness': avg_val_fitness,
                'best_train_return': best_train['return'],
                'best_val_return': best_val['return']
            })
            
            # Early stopping
            if self.generations_without_improvement >= self.early_stop_patience:
                print(f"\nEarly stopping triggered after {gen + 1} generations")
                break
            
            # Create next generation
            if gen < generations - 1:
                population = self.create_next_generation(evaluated)
        
        return self.best_params, self.history
    
    def plot_evolution(self):
        """Plot optimization progress"""
        df = pd.DataFrame(self.history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Fitness evolution
        ax1.plot(df['generation'], df['best_train_fitness'], 'b-', label='Train', linewidth=2)
        ax1.plot(df['generation'], df['best_val_fitness'], 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Best Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average fitness
        ax2.plot(df['generation'], df['avg_train_fitness'], 'b--', label='Avg Train', alpha=0.7)
        ax2.plot(df['generation'], df['avg_val_fitness'], 'r--', label='Avg Val', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Average Population Fitness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Return evolution
        ax3.plot(df['generation'], df['best_train_return'] * 100, 'g-', label='Train', linewidth=2)
        ax3.plot(df['generation'], df['best_val_return'] * 100, 'orange', label='Validation', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Return %')
        ax3.set_title('Best Return Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Overfitting detection
        ax4.plot(df['generation'], 
                 df['best_train_fitness'] - df['best_val_fitness'], 
                 'purple', linewidth=2)
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Train - Val Fitness Gap')
        ax4.set_title('Overfitting Monitor')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def print_optimized_params(params: GeneticParams):
    """Print optimized parameters vs defaults"""
    print("\n" + "="*60)
    print("OPTIMIZED PARAMETERS (with validation)")
    print("="*60)
    
    defaults = GeneticParams()
    
    print("\nEntry Thresholds:")
    print(f"  Momentum: {params.momentum_entry:.2f} (default: {defaults.momentum_entry:.2f})")
    print(f"  Post-massive: {params.post_massive_min:.2f}-{params.post_massive_max:.2f} "
          f"(default: {defaults.post_massive_min:.2f}-{defaults.post_massive_max:.2f})")
    print(f"  Hot streak: {params.hot_streak_4_min:.2f}-{params.hot_streak_4_max:.2f} "
          f"(default: {defaults.hot_streak_4_min:.2f}-{defaults.hot_streak_4_max:.2f})")
    
    print("\nTargets:")
    print(f"  Momentum: {params.momentum_target_mult:.1f}x (default: {defaults.momentum_target_mult:.1f}x)")
    print(f"  Post-massive: {params.post_massive_target_mult:.2f}x (default: {defaults.post_massive_target_mult:.2f}x)")
    print(f"  Hot streak 4: {params.hot_streak_4_target:.1f}x (default: {defaults.hot_streak_4_target:.1f}x)")
    print(f"  Hot streak 5: {params.hot_streak_5_target:.1f}x (default: {defaults.hot_streak_5_target:.1f}x)")
    
    print("\nStop Losses:")
    print(f"  Momentum: {params.momentum_stop:.2f} (default: {defaults.momentum_stop:.2f})")
    print(f"  Post-massive: {params.post_massive_stop:.2f} (default: {defaults.post_massive_stop:.2f})")
    print(f"  Hot streak: {params.hot_streak_stop:.2f} (default: {defaults.hot_streak_stop:.2f})")
    
    print("\nBase Bet Sizes:")
    print(f"  Momentum: {params.momentum_base_bet*100:.2f}% (default: {defaults.momentum_base_bet*100:.2f}%)")
    print(f"  Post-massive: {params.post_massive_base_bet*100:.2f}% (default: {defaults.post_massive_base_bet*100:.2f}%)")
    print(f"  Hot streak: {params.hot_streak_base_bet*100:.2f}% (default: {defaults.hot_streak_base_bet*100:.2f}%)")
    print(f"  Hot streak 5: {params.hot_streak_5_base_bet*100:.2f}% (default: {defaults.hot_streak_5_base_bet*100:.2f}%)")

def load_data(folder='clean_total_rounds'):
    """Fixed data load - loads the correct round_XXXX.csv files"""
    print(f"Loading from {folder}...")
    start = time.time()
    
    rounds = {}
    base = Path(folder)
    if not base.exists():
        base = Path('total_rounds')
    
    count = 0
    for subfolder in sorted(base.iterdir()):
        if subfolder.is_dir():
            # Look for round_XXXX.csv specifically, NOT processed.csv
            round_name = subfolder.name
            round_csv = subfolder / f"{round_name}.csv"
            
            if round_csv.exists():
                try:
                    df = pd.read_csv(round_csv)
                    
                    # Debug first file
                    if count == 0:
                        print(f"First CSV: {round_csv.name}")
                        print(f"Columns: {list(df.columns)}")
                    
                    # Get multiplier column
                    if 'multiplier' in df.columns:
                        rounds[round_name] = df['multiplier'].values
                        count += 1
                    else:
                        print(f"WARNING: No multiplier column in {round_csv}")
                        print(f"  Columns: {list(df.columns)}")
                        
                except Exception as e:
                    print(f"Error loading {round_csv}: {e}")
            else:
                # Fallback: try to find any CSV that matches round pattern
                round_csvs = list(subfolder.glob(f"{round_name}*.csv"))
                if round_csvs:
                    try:
                        df = pd.read_csv(round_csvs[0])
                        if 'multiplier' in df.columns:
                            rounds[round_name] = df['multiplier'].values
                            count += 1
                    except Exception as e:
                        print(f"Error loading {round_csvs[0]}: {e}")
    
    print(f"Loaded {count} rounds in {time.time()-start:.1f}s")
    return rounds

def run_final_comparison(all_rounds, optimized_params):
    """Compare default vs optimized on train/val/test"""
    rounds_list = sorted(all_rounds.items())
    n_rounds = len(rounds_list)
    
    # Same splits as optimizer
    train_end = int(n_rounds * 0.6)
    val_end = int(n_rounds * 0.8)
    
    train_rounds = rounds_list[:train_end]
    val_rounds = rounds_list[train_end:val_end]
    test_rounds = rounds_list[val_end:]
    
    print("\n" + "="*60)
    print("FINAL COMPARISON: Default vs Optimized")
    print("="*60)
    
    # Test both parameter sets
    default_params = GeneticParams()
    
    for name, params in [("Default", default_params), ("Optimized", optimized_params)]:
        print(f"\n{name} Parameters:")
        
        # Run on all three sets
        train_results = fast_backtest(train_rounds, params)
        val_results = fast_backtest(val_rounds, params)
        test_results = fast_backtest(test_rounds, params)
        
        print(f"  Train: Return={train_results['return']*100:.1f}%, DD={train_results['max_dd']*100:.1f}%, Trades={train_results['n_trades']}")
        print(f"  Val:   Return={val_results['return']*100:.1f}%, DD={val_results['max_dd']*100:.1f}%, Trades={val_results['n_trades']}")
        print(f"  TEST:  Return={test_results['return']*100:.1f}%, DD={test_results['max_dd']*100:.1f}%, Trades={test_results['n_trades']}")
        
        # Check for overfitting
        train_val_gap = abs(train_results['return'] - val_results['return'])
        if train_val_gap > 0.3:  # 30% gap
            print(f"  ⚠️  Warning: Large train-val gap ({train_val_gap*100:.1f}%) suggests overfitting")

# Main execution
if __name__ == "__main__":
    # Load data
    rounds = load_data()
    if not rounds:
        print("No data found!")
        exit()
    
    # Create optimizer with proper data splits
    optimizer = RobustGeneticOptimizer(
        all_rounds=rounds,
        train_ratio=0.6,  # 60% for training
        val_ratio=0.2,    # 20% for validation
        population_size=40,  # Slightly smaller for faster convergence
        elite_size=8
    )
    
    # Run optimization
    best_params, history = optimizer.optimize(generations=30)
    
    # Show evolution
    optimizer.plot_evolution()
    
    # Print optimized parameters
    print_optimized_params(best_params)
    
    # Final comparison on all sets including TEST
    run_final_comparison(rounds, best_params)
    
    # Save parameters
    params_dict = {field: getattr(best_params, field) for field in best_params.__dataclass_fields__}
    with open('robust_optimized_params.json', 'w') as f:
        json.dump(params_dict, f, indent=2)
    print("\nSaved optimized parameters to robust_optimized_params.json")