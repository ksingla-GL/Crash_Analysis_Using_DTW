#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genetic Algorithm for Optimizing Crash Game Betting Strategy Parameters.

This script uses a genetic algorithm to find the optimal set of parameters for a
multi-strategy betting bot. It employs a robust train/validation/test split to
prevent overfitting and ensures the resulting strategy is generalizable.

The algorithm optimizes parameters such as:
- Entry/exit thresholds for different strategies (Momentum, Post-Massive, etc.)
- Base bet sizes as a percentage of capital
- Dynamic scaling factors based on performance and drawdowns

The final output is a JSON file containing the best-performing parameter set
found during the optimization process.

@author: Kshitij
"""

# --- Imports ---
import json
import random
import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
CONFIG = {
    "DATA_FOLDER": "total_rounds",
    "TRAIN_RATIO": 0.6,
    "VALIDATION_RATIO": 0.2,
    "POPULATION_SIZE": 40,
    "ELITE_SIZE": 8,
    "GENERATIONS": 30,
    "EARLY_STOP_PATIENCE": 10,
    "ENTRY_TICK": 20,
    "OUTPUT_PARAMS_FILE": "robust_optimized_params.json"
}


# --- Parameter and Bot Definitions ---

@dataclass
class GeneticParams:
    """
    Defines the complete set of parameters (the "genes") for a single
    bot strategy that the genetic algorithm will optimize.
    """
    # Entry thresholds
    momentum_entry: float = 2.0
    post_massive_min: float = 1.05
    post_massive_max: float = 1.5
    hot_streak_4_min: float = 1.0
    hot_streak_4_max: float = 2.0

    # Targets
    momentum_target_mult: float = 10.0
    post_massive_target_mult: float = 1.5
    hot_streak_4_target: float = 3.0
    hot_streak_5_target: float = 5.0

    # Stop losses
    momentum_stop: float = 0.8
    post_massive_stop: float = 0.9
    hot_streak_stop: float = 0.85

    # Base bet sizes (% of capital)
    momentum_base_bet: float = 0.008
    post_massive_base_bet: float = 0.015
    hot_streak_base_bet: float = 0.010
    hot_streak_5_base_bet: float = 0.014

    # Other strategy parameters
    massive_ago_limit: int = 3
    hot_streak_count: int = 4

    def mutate(self, mutation_rate=0.1, conservative=True):
        """
        Creates a new, mutated version of the parameters.

        Args:
            mutation_rate (float): The probability of any single gene mutating.
            conservative (bool): If True, applies smaller mutation ranges.

        Returns:
            GeneticParams: A new instance with mutated parameters.
        """
        mutated = GeneticParams()

        for field in fields(self):
            value = getattr(self, field.name)
            if random.random() < mutation_rate:
                # Determine mutation range based on conservatism
                range_map = {
                    'conservative': {
                        'entry': (0.9, 1.1), 'target': (0.85, 1.15),
                        'stop': (0.95, 1.05), 'bet': (0.85, 1.15)
                    },
                    'aggressive': {
                        'entry': (0.8, 1.2), 'target': (0.7, 1.3),
                        'stop': (0.9, 1.1), 'bet': (0.7, 1.3)
                    }
                }
                ranges = range_map['conservative' if conservative else 'aggressive']
                
                # Apply mutation based on field type
                if 'entry' in field.name or 'min' in field.name or 'max' in field.name:
                    value *= random.uniform(*ranges['entry'])
                elif 'target' in field.name:
                    value *= random.uniform(*ranges['target'])
                elif 'stop' in field.name:
                    value *= random.uniform(*ranges['stop'])
                elif 'bet' in field.name:
                    value *= random.uniform(*ranges['bet'])
                elif field.name == 'massive_ago_limit':
                    value = max(2, min(5, value + random.randint(-1, 1)))
                elif field.name == 'hot_streak_count':
                    value = random.choice([3, 4, 5])

            # Apply constraints to keep parameters within a reasonable range
            if 'stop' in field.name:
                value = max(0.5, min(0.95, value))
            elif 'bet' in field.name:
                value = max(0.002, min(0.03, value))
            elif 'momentum_entry' in field.name:
                value = max(1.5, min(3.0, value))
            elif 'target_mult' in field.name:
                value = max(1.2, min(20.0, value))

            setattr(mutated, field.name, value)
        return mutated

    def crossover(self, other: "GeneticParams") -> "GeneticParams":
        """
        Performs uniform crossover between two parents to create a child.

        Args:
            other (GeneticParams): The second parent.

        Returns:
            GeneticParams: A new child instance with mixed genes.
        """
        child = GeneticParams()
        for field in fields(self):
            # 50/50 chance to inherit the gene from either parent
            source_parent = self if random.random() < 0.5 else other
            setattr(child, field.name, getattr(source_parent, field.name))
        return child
        
    def regularization_penalty(self) -> float:
        """
        Calculates a penalty for parameters that deviate significantly from
        the defaults. This helps prevent overfitting to extreme values.

        Returns:
            float: The calculated penalty value.
        """
        penalty = 0.0
        defaults = GeneticParams()
        for field in fields(self):
            default_val = getattr(defaults, field.name)
            current_val = getattr(self, field.name)
            if isinstance(default_val, (int, float)):
                deviation = abs(current_val - default_val) / (default_val + 1e-6)
                penalty += deviation * 0.01  # Small penalty factor
        return penalty

class OptimizedBot:
    """
    A betting bot that operates using a given set of GeneticParams.
    This class is used within the backtest to evaluate a parameter set.
    """
    def __init__(self, params: GeneticParams, capital: float = 1000.0):
        self.params = params
        self.capital = capital
        self.init_capital = capital
        self.peak_capital = capital
        self.entry_tick = CONFIG["ENTRY_TICK"]

        # State tracking
        self.massive_ago = 999
        self.recent_round_maxes = deque(maxlen=5)
        self.trades_history = [] # Simplified for speed
        
        # Performance tracking by strategy
        self.momentum_pnls = deque(maxlen=20)
        self.post_massive_pnls = deque(maxlen=30)
        self.hot_streak_pnls = deque(maxlen=20)
        
        # Overall stats
        self.n_trades = 0
        self.n_wins = 0

    def decide_trade(self, entry_price: float) -> Optional[Tuple[str, float, float]]:
        """
        Determines if a trade should be made based on the current entry price
        and the bot's internal state, using its genetic parameters.

        Args:
            entry_price (float): The price at the potential entry tick.

        Returns:
            Optional[Tuple[str, float, float]]: A tuple of
            (strategy_name, target_price, stop_ratio) if a trade is warranted,
            otherwise None.
        """
        # Momentum Strategy
        if entry_price >= self.params.momentum_entry:
            target = entry_price * self.params.momentum_target_mult
            return 'momentum', target, self.params.momentum_stop

        # Skip rounds that already crashed
        if entry_price < 1.0:
            return None

        # Post-Massive-Crash Strategy
        if (self.massive_ago <= self.params.massive_ago_limit and
            self.params.post_massive_min <= entry_price < self.params.post_massive_max):
            target = entry_price * self.params.post_massive_target_mult
            return 'post_massive', target, self.params.post_massive_stop

        # Hot Streaks Strategy
        if len(self.recent_round_maxes) >= self.params.hot_streak_count:
            streak_count = sum(1 for max_val in self.recent_round_maxes if max_val >= 2.0)
            is_in_range = self.params.hot_streak_4_min <= entry_price < self.params.hot_streak_4_max
            
            if streak_count >= 5 and is_in_range:
                return 'hot_streak_5', self.params.hot_streak_5_target, self.params.hot_streak_stop
            elif streak_count >= self.params.hot_streak_count and is_in_range:
                return 'hot_streak', self.params.hot_streak_4_target, self.params.hot_streak_stop

        return None

    def calculate_bet_size(self, strat_type: str) -> float:
        """
        Calculates the bet size for a given strategy, applying dynamic scaling
        based on performance and drawdown.

        Args:
            strat_type (str): The name of the strategy being used.

        Returns:
            float: The calculated bet amount.
        """
        # Select base bet and PnL history based on strategy
        strat_map = {
            'momentum': (self.params.momentum_base_bet, self.momentum_pnls),
            'post_massive': (self.params.post_massive_base_bet, self.post_massive_pnls),
            'hot_streak': (self.params.hot_streak_base_bet, self.hot_streak_pnls),
            'hot_streak_5': (self.params.hot_streak_5_base_bet, self.hot_streak_pnls),
        }
        base_bet_pct, recent_pnls = strat_map[strat_type]
        
        # Apply multipliers
        multiplier = 1.0
        if len(recent_pnls) > 10:
            win_rate = sum(1 for pnl in recent_pnls if pnl > 0) / len(recent_pnls)
            if win_rate > 0.35: multiplier *= 1.3
            elif win_rate < 0.15: multiplier *= 0.7

        # Apply drawdown protection
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if drawdown > 0.05:
            multiplier *= max(1.0 - drawdown, 0.5) # Gradual reduction

        # Calculate final bet and apply capital constraints
        bet = self.capital * base_bet_pct * multiplier
        return min(bet, self.capital * 0.05) if bet >= 1.0 else 0.0


# --- Backtesting and Optimization ---

def fast_backtest(rounds_data: List[Tuple[str, np.ndarray]], params: GeneticParams) -> Dict:
    """
    A simplified, fast backtest engine to quickly evaluate the fitness
    of a single GeneticParams instance.

    Args:
        rounds_data (List): The dataset (train, val, or test) to backtest on.
        params (GeneticParams): The parameter set to evaluate.

    Returns:
        Dict: A dictionary of performance metrics (return, drawdown, fitness, etc.).
    """
    bot = OptimizedBot(params)
    
    for i in range(len(rounds_data) - 1):
        round_id, data = rounds_data[i]
        next_round_id, next_data = rounds_data[i+1]

        # 1. Update bot state with results of the completed round
        if data is not None and len(data) > 0:
            round_max = np.max(data)
            bot.recent_round_maxes.append(round_max)
            bot.massive_ago = 0 if round_max >= 10.0 else bot.massive_ago + 1
        
        # 2. Check if we can trade the *next* round
        if next_data is None or len(next_data) <= bot.entry_tick:
            continue
            
        entry_price = next_data[bot.entry_tick]
        trade_decision = bot.decide_trade(entry_price)
        
        if not trade_decision:
            continue
        
        strat_type, target, stop_ratio = trade_decision
        
        # 3. Size the bet and execute
        bet = bot.calculate_bet_size(strat_type)
        if bet <= 0:
            continue
        
        bot.n_trades += 1
        bot.capital -= bet
        
        # 4. Simulate trade outcome
        exit_price = 0.0
        for tick_price in next_data[bot.entry_tick + 1:]:
            if tick_price >= target or (tick_price / entry_price) <= stop_ratio:
                exit_price = tick_price
                break
        
        pnl = (bet * (exit_price / entry_price)) - bet if entry_price > 0 else -bet
        bot.capital += (bet + pnl)
        
        # 5. Record results
        if pnl > 0: bot.n_wins += 1
        bot.peak_capital = max(bot.peak_capital, bot.capital)
        
        # Update strategy-specific PnLs
        if strat_type == 'momentum': bot.momentum_pnls.append(pnl)
        elif strat_type == 'post_massive': bot.post_massive_pnls.append(pnl)
        else: bot.hot_streak_pnls.append(pnl)

    # 6. Calculate final fitness metrics
    total_return = (bot.capital - bot.init_capital) / bot.init_capital
    max_dd = (bot.peak_capital - bot.capital) / bot.peak_capital if bot.peak_capital > 0 else 0
    win_rate = bot.n_wins / bot.n_trades if bot.n_trades > 0 else 0
    
    # Combined fitness score: reward return, penalize drawdown
    fitness = total_return * (1 - max_dd**0.5) # Square root to lessen DD impact slightly
    
    return {
        'return': total_return, 'max_dd': max_dd, 'win_rate': win_rate,
        'n_trades': bot.n_trades, 'final_capital': bot.capital, 'fitness': fitness
    }

class RobustGeneticOptimizer:
    """
    Manages the genetic algorithm process, including data splitting,
    evolutionary cycles, and overfitting checks.
    """
    def __init__(self, all_rounds: Dict, config: Dict):
        self.config = config
        self._split_data(all_rounds)
        
        self.population_size = config["POPULATION_SIZE"]
        self.elite_size = config["ELITE_SIZE"]
        self.best_params: Optional[GeneticParams] = None
        self.best_val_fitness = -float('inf')
        self.history = []
        self.generations_without_improvement = 0

    def _split_data(self, all_rounds: Dict):
        """Splits data into training, validation, and test sets."""
        rounds_list = sorted(all_rounds.items())
        n = len(rounds_list)
        train_end = int(n * self.config["TRAIN_RATIO"])
        val_end = train_end + int(n * self.config["VALIDATION_RATIO"])
        
        self.train_rounds = rounds_list[:train_end]
        self.val_rounds = rounds_list[train_end:val_end]
        self.test_rounds = rounds_list[val_end:]
        
        print(f"Data split: Train={len(self.train_rounds)}, Val={len(self.val_rounds)}, Test={len(self.test_rounds)}")

    def create_initial_population(self) -> List[GeneticParams]:
        """Creates a diverse starting population."""
        population = [GeneticParams()]  # Start with defaults
        while len(population) < self.population_size:
            # Create variety by mutating the default params
            population.append(GeneticParams().mutate(mutation_rate=0.4, conservative=True))
        return population

    def evaluate_population(self, population: List[GeneticParams]) -> List[Tuple[GeneticParams, Dict, Dict]]:
        """Evaluates each individual on both training and validation sets."""
        results = []
        for i, params in enumerate(population):
            if (i + 1) % 10 == 0: print(f"  Evaluating individual {i + 1}/{len(population)}...")
            
            train_metrics = fast_backtest(self.train_rounds, params)
            val_metrics = fast_backtest(self.val_rounds, params)
            
            # Apply regularization penalty to fitness to discourage extreme values
            reg_penalty = params.regularization_penalty()
            train_metrics['fitness'] -= reg_penalty
            val_metrics['fitness'] -= reg_penalty
            
            results.append((params, train_metrics, val_metrics))
        
        # IMPORTANT: Sort by VALIDATION fitness to select the best generalists
        return sorted(results, key=lambda x: x[2]['fitness'], reverse=True)

    def create_next_generation(self, evaluated_pop: List[Tuple[GeneticParams, Dict, Dict]]) -> List[GeneticParams]:
        """Breeds a new generation from the evaluated population."""
        # 1. Elitism: Keep the best performers
        next_gen = [ind[0] for ind in evaluated_pop[:self.elite_size]]
        
        # 2. Breeding: Fill the rest with children of good performers
        while len(next_gen) < self.population_size:
            # Tournament selection is a robust way to pick parents
            parent1 = self.tournament_select(evaluated_pop)
            parent2 = self.tournament_select(evaluated_pop)
            child = parent1.crossover(parent2)
            
            # Apply mutation to introduce new genetic material
            mutation_rate = 0.2 * (1 - len(self.history) / self.config["GENERATIONS"]) # Decaying rate
            if random.random() < mutation_rate:
                child = child.mutate(conservative=True)
            
            next_gen.append(child)
            
        return next_gen
    
    def tournament_select(self, evaluated_pop: List[Tuple[GeneticParams, Dict, Dict]], size=3) -> GeneticParams:
        """Selects the best individual from a random subsample."""
        top_half = evaluated_pop[:len(evaluated_pop)//2]
        tournament = random.sample(top_half, min(size, len(top_half)))
        # Winner is the one with the best validation fitness
        return max(tournament, key=lambda x: x[2]['fitness'])[0]

    def optimize(self) -> Tuple[Optional[GeneticParams], List[Dict]]:
        """Runs the main optimization loop for a set number of generations."""
        print(f"\nStarting robust genetic optimization for {self.config['GENERATIONS']} generations...")
        population = self.create_initial_population()

        for gen in range(self.config["GENERATIONS"]):
            print(f"\n--- Generation {gen + 1}/{self.config['GENERATIONS']} ---")
            
            evaluated = self.evaluate_population(population)
            
            best_ind, best_train, best_val = evaluated[0]
            
            # Track progress and check for early stopping
            if best_val['fitness'] > self.best_val_fitness:
                self.best_val_fitness = best_val['fitness']
                self.best_params = best_ind
                self.generations_without_improvement = 0
                print(f"  New best validation fitness found: {self.best_val_fitness:.4f}")
            else:
                self.generations_without_improvement += 1

            # Logging
            print(f"  Best Val Fitness: {best_val['fitness']:.4f} (Return: {best_val['return']*100:.1f}%)")
            print(f"  Best Train Fitness: {best_train['fitness']:.4f} (Return: {best_train['return']*100:.1f}%)")
            print(f"  Generations without improvement: {self.generations_without_improvement}")
            self.history.append({'gen': gen + 1, 'best_val_fitness': best_val['fitness'], 'best_train_fitness': best_train['fitness']})
            
            if self.generations_without_improvement >= self.config["EARLY_STOP_PATIENCE"]:
                print(f"\nEarly stopping triggered after {gen + 1} generations.")
                break
                
            population = self.create_next_generation(evaluated)
            
        return self.best_params, self.history


# --- Data Loading and Visualization ---

def load_data(folder: str) -> Dict[str, np.ndarray]:
    """Loads all round data CSVs from a specified folder."""
    print(f"Loading round data from '{folder}'...")
    start_time = time.time()
    rounds = {}
    base_path = Path(folder)
    
    if not base_path.exists():
        print(f"Error: Data directory '{folder}' not found.")
        return {}

    for subfolder in sorted(base_path.iterdir()):
        if not subfolder.is_dir(): continue
        
        round_csv = subfolder / f"{subfolder.name}.csv"
        if round_csv.exists():
            try:
                df = pd.read_csv(round_csv)
                if 'multiplier' in df.columns:
                    rounds[subfolder.name] = df['multiplier'].values
            except Exception as e:
                print(f"Error loading {round_csv}: {e}")

    print(f"Loaded {len(rounds)} rounds in {time.time() - start_time:.2f} seconds.")
    return rounds

def plot_evolution(history: List[Dict]):
    """Plots the fitness evolution over generations."""
    if not history: return
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(12, 7))
    plt.plot(df['gen'], df['best_train_fitness'], 'b-', label='Best Train Fitness', linewidth=2)
    plt.plot(df['gen'], df['best_val_fitness'], 'r-', label='Best Validation Fitness', linewidth=2)
    
    # Check for overfitting
    overfit_gap = df['best_train_fitness'] - df['best_val_fitness']
    plt.fill_between(df['gen'], df['best_val_fitness'], df['best_train_fitness'], 
                     where=overfit_gap > 0, color='blue', alpha=0.1, label='Overfitting Gap')
    
    plt.title('Genetic Algorithm Fitness Evolution', fontsize=16)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def print_final_comparison(all_rounds: Dict, best_params: GeneticParams, config: Dict):
    """Compares the default vs. optimized parameters on all data splits."""
    print("\n" + "="*60)
    print("FINAL COMPARISON: Default vs. Optimized Parameters")
    print("="*60)

    optimizer = RobustGeneticOptimizer(all_rounds, config) # To get the same data splits
    param_sets = {"Default": GeneticParams(), "Optimized": best_params}

    for name, params in param_sets.items():
        print(f"\n--- {name} Strategy Results ---")
        train_res = fast_backtest(optimizer.train_rounds, params)
        val_res = fast_backtest(optimizer.val_rounds, params)
        test_res = fast_backtest(optimizer.test_rounds, params)
        
        print(f"  Train Set: Return={train_res['return']*100:6.1f}%, DD={train_res['max_dd']*100:5.1f}%, Trades={train_res['n_trades']}")
        print(f"  Val.  Set: Return={val_res['return']*100:6.1f}%, DD={val_res['max_dd']*100:5.1f}%, Trades={val_res['n_trades']}")
        print(f"  Test  Set: Return={test_res['return']*100:6.1f}%, DD={test_res['max_dd']*100:5.1f}%, Trades={test_res['n_trades']}")


# --- Main Execution ---

def main():
    """Main script execution function."""
    # 1. Load data
    rounds = load_data(CONFIG["DATA_FOLDER"])
    if not rounds:
        return

    # 2. Set up and run the optimizer
    optimizer = RobustGeneticOptimizer(rounds, CONFIG)
    best_params, history = optimizer.optimize()

    # 3. Analyze and save results
    if best_params:
        plot_evolution(history)
        print_final_comparison(rounds, best_params, CONFIG)
        
        # Save the best parameters to a file
        params_dict = {f.name: getattr(best_params, f.name) for f in fields(best_params)}
        with open(CONFIG["OUTPUT_PARAMS_FILE"], 'w') as f:
            json.dump(params_dict, f, indent=4)
        print(f"\n✅ Saved optimized parameters to '{CONFIG['OUTPUT_PARAMS_FILE']}'")
    else:
        print("\nOptimization did not find a suitable parameter set.")

if __name__ == "__main__":
    main()