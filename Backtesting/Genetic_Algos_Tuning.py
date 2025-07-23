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
            target = entry_price *