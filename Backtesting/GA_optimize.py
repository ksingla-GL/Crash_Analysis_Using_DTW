import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

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
    
    def mutate(self, mutation_rate=0.1):
        """Mutate parameters with given probability"""
        mutated = GeneticParams()
        
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if random.random() < mutation_rate:
                # Different mutation ranges for different types
                if 'entry' in field or 'min' in field or 'max' in field:
                    # Entry points: ±20%
                    value *= random.uniform(0.8, 1.2)
                elif 'target' in field:
                    # Targets: ±30%
                    value *= random.uniform(0.7, 1.3)
                elif 'stop' in field:
                    # Stops: ±10%
                    value *= random.uniform(0.9, 1.1)
                elif 'bet' in field:
                    # Bet sizes: ±30%
                    value *= random.uniform(0.7, 1.3)
                elif field == 'massive_ago_limit':
                    # Integer: ±1
                    value = max(1, value + random.randint(-1, 1))
                elif field == 'hot_streak_count':
                    # Integer: 3-5
                    value = random.randint(3, 5)
            
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

def fast_backtest(rounds, params: GeneticParams) -> Dict:
    """Fast backtest for genetic algorithm"""
    bot = OptimizedBot(params)
    sorted_rounds = list(rounds.items())
    
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
    max_dd = (bot.peak_capital - bot.capital) / bot.peak_capital
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

class GeneticOptimizer:
    """Genetic algorithm for parameter optimization"""
    
    def __init__(self, rounds_data, population_size=50, elite_size=10):
        self.rounds = rounds_data
        self.population_size = population_size
        self.elite_size = elite_size
        self.generation = 0
        self.best_params = None
        self.best_fitness = -float('inf')
        self.history = []
        
    def create_initial_population(self) -> List[GeneticParams]:
        """Create random initial population"""
        population = []
        
        # Add default params
        population.append(GeneticParams())
        
        # Add random variations
        for _ in range(self.population_size - 1):
            params = GeneticParams()
            # Mutate heavily for diversity
            for _ in range(3):
                params = params.mutate(mutation_rate=0.5)
            population.append(params)
        
        return population
    
    def evaluate_population(self, population: List[GeneticParams]) -> List[Tuple[GeneticParams, Dict]]:
        """Evaluate fitness of all individuals"""
        results = []
        
        for i, params in enumerate(population):
            if i % 10 == 0:
                print(f"  Evaluating individual {i+1}/{len(population)}...")
            
            metrics = fast_backtest(self.rounds, params)
            results.append((params, metrics))
        
        # Sort by fitness
        results.sort(key=lambda x: x[1]['fitness'], reverse=True)
        return results
    
    def create_next_generation(self, evaluated_pop: List[Tuple[GeneticParams, Dict]]) -> List[GeneticParams]:
        """Create next generation via selection, crossover, and mutation"""
        next_gen = []
        
        # Keep elite
        for i in range(self.elite_size):
            next_gen.append(evaluated_pop[i][0])
        
        # Fill rest with crossover and mutation
        while len(next_gen) < self.population_size:
            # Tournament selection
            parent1 = self.tournament_select(evaluated_pop)
            parent2 = self.tournament_select(evaluated_pop)
            
            # Crossover
            child = parent1.crossover(parent2)
            
            # Mutation
            if random.random() < 0.2:  # 20% mutation rate
                child = child.mutate()
            
            next_gen.append(child)
        
        return next_gen
    
    def tournament_select(self, evaluated_pop: List[Tuple[GeneticParams, Dict]], 
                         tournament_size=3) -> GeneticParams:
        """Tournament selection"""
        tournament = random.sample(evaluated_pop[:20], min(tournament_size, 20))  # Top 20 only
        winner = max(tournament, key=lambda x: x[1]['fitness'])
        return winner[0]
    
    def optimize(self, generations=50):
        """Run genetic algorithm"""
        print(f"Starting genetic optimization with {self.population_size} individuals...")
        
        # Initial population
        population = self.create_initial_population()
        
        for gen in range(generations):
            print(f"\nGeneration {gen + 1}/{generations}")
            
            # Evaluate
            evaluated = self.evaluate_population(population)
            
            # Track best
            best_individual = evaluated[0]
            if best_individual[1]['fitness'] > self.best_fitness:
                self.best_fitness = best_individual[1]['fitness']
                self.best_params = best_individual[0]
            
            # Log progress
            avg_fitness = np.mean([x[1]['fitness'] for x in evaluated])
            avg_return = np.mean([x[1]['return'] for x in evaluated])
            
            print(f"  Best fitness: {best_individual[1]['fitness']:.4f}")
            print(f"  Best return: {best_individual[1]['return']*100:.1f}%")
            print(f"  Best DD: {best_individual[1]['max_dd']*100:.1f}%")
            print(f"  Avg fitness: {avg_fitness:.4f}")
            
            self.history.append({
                'generation': gen + 1,
                'best_fitness': best_individual[1]['fitness'],
                'best_return': best_individual[1]['return'],
                'avg_fitness': avg_fitness,
                'avg_return': avg_return
            })
            
            # Create next generation
            if gen < generations - 1:
                population = self.create_next_generation(evaluated)
        
        return self.best_params, self.history
    
    def plot_evolution(self):
        """Plot optimization progress"""
        df = pd.DataFrame(self.history)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Fitness evolution
        ax1.plot(df['generation'], df['best_fitness'], 'b-', label='Best', linewidth=2)
        ax1.plot(df['generation'], df['avg_fitness'], 'r--', label='Average', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Return evolution
        ax2.plot(df['generation'], df['best_return'] * 100, 'g-', label='Best', linewidth=2)
        ax2.plot(df['generation'], df['avg_return'] * 100, 'orange', label='Average', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Return %')
        ax2.set_title('Return Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def print_optimized_params(params: GeneticParams):
    """Print optimized parameters vs defaults"""
    print("\n" + "="*60)
    print("OPTIMIZED PARAMETERS")
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
    
    print("\nOther:")
    print(f"  Massive ago limit: {params.massive_ago_limit} (default: {defaults.massive_ago_limit})")
    print(f"  Hot streak count: {params.hot_streak_count} (default: {defaults.hot_streak_count})")

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

# Main execution
if __name__ == "__main__":
    # Load data
    rounds = load_data()
    if not rounds:
        print("No data found!")
        exit()
    
    # Create optimizer
    optimizer = GeneticOptimizer(
        rounds_data=rounds,
        population_size=50,  # Number of parameter sets per generation
        elite_size=10  # Top performers to keep
    )
    
    # Run optimization
    best_params, history = optimizer.optimize(generations=30)
    
    # Show results
    optimizer.plot_evolution()
    print_optimized_params(best_params)
    
    # Run final detailed backtest with best params
    print("\nRunning detailed backtest with optimized parameters...")
    final_metrics = fast_backtest(rounds, best_params)
    
    print(f"\nFINAL RESULTS:")
    print(f"Return: {final_metrics['return']*100:.1f}%")
    print(f"Max Drawdown: {final_metrics['max_dd']*100:.1f}%")
    print(f"Win Rate: {final_metrics['win_rate']*100:.1f}%")
    print(f"Total Trades: {final_metrics['n_trades']}")
    print(f"Sharpe Ratio: {final_metrics['sharpe']:.2f}")
    print(f"Final Capital: ${final_metrics['final_capital']:,.2f}")
    
    # Save best parameters
    import json
    params_dict = {field: getattr(best_params, field) for field in best_params.__dataclass_fields__}
    with open('optimized_params.json', 'w') as f:
        json.dump(params_dict, f, indent=2)
    print("\nSaved optimized parameters to optimized_params.json")