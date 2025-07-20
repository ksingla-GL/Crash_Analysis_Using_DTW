import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import time


class ImprovedMomentumBot:
    """
    my crash game bot - trying to beat the house
    
    main strats:
    - momentum plays when price already pumping (>2x)
    - post massive wins the game tends to cool down  
    - hot streaks when games been hitting 2x+ multiple times
    
    bet sizing is kinda complex but basically:
    - bet more when winning
    - bet less when losing  
    - dont blow up account during drawdowns
    """
    
    def __init__(self, capital=1000.0):
        self.capital = capital
        self.init_capital = capital
        self.trades = []
        
        # tracking stuff
        self.last_max = None
        self.massive_ago = 999  # how long since we saw a 10x+
        self.recent_pnls = deque(maxlen=10)  
        self.peak_capital = capital
        
        # keep track of recent rounds for streak detection
        self.recent_round_maxes = deque(maxlen=5)  
        
        # separate tracking for each strat (found this works better)
        self.momentum_pnls = deque(maxlen=20)
        self.post_massive_pnls = deque(maxlen=30)
        self.hot_streak_pnls = deque(maxlen=20)  
        
        # basic stats
        self.n_trades = 0
        self.n_wins = 0
        self.skipped = 0
        
        # per strategy stats
        self.momentum_trades = 0
        self.momentum_wins = 0
        self.post_massive_trades = 0  
        self.post_massive_wins = 0
        self.hot_streak_trades = 0
        self.hot_streak_wins = 0
        
        # bet sizes - found these through trial and error thanks to genetic algos
        self.momentum_base = 0.008  # 0.8% - small cuz low winrate
        self.post_massive_base = 0.015  # 1.5% - bigger cuz wins more often
        
        self.entry_tick = 20  # wait 20 ticks before entering
        
    def should_trade(self, entry_price, round_data):
        """figure out if we should trade this round"""
        
        # momentum play - when already mooning
        if entry_price >= 2.0:
            return 'momentum', 20.0, 0.8  # target 20x, stop at 0.8
        
        # skip if already crashed
        if entry_price < 1.0:
            self.skipped += 1
            return None, 0, 0
        
        # post massive cooling - noticed this pattern after big wins
        # tweaked these values a bunch
        if self.massive_ago <= 3 and entry_price >= 1.1 and entry_price < 1.6:
            return 'post_massive', entry_price * 1.4, 0.85  # more realistic target
        
        # hot streak detection - when games been popping off
        if len(self.recent_round_maxes) >= 3:
            # count how many hit 2x recently
            streak_count = sum(1 for max_val in self.recent_round_maxes if max_val >= 2.0)
            
            # super hot - 5 in a row!!
            if streak_count >= 5 and 1.0 <= entry_price < 2.0:
                return 'hot_streak_5', 5.0, 0.85  # go for 5x
                
            # pretty hot - 4 rounds
            elif streak_count >= 4 and 1.0 <= entry_price < 2.0:
                return 'hot_streak', 3.0, 0.85  # target 3x
                
            # warm - just 3 rounds
            elif streak_count >= 3 and 1.1 <= entry_price < 2.0:
                return 'hot_streak_3', 2.0, 0.88  # conservative 2x target
        
        self.skipped += 1
        return None, 0, 0
    
    def get_bet_size(self, strat_type):
        """calculate how much to bet - this is where the magic happens"""
        
        # base bet depends on strategy
        if strat_type == 'momentum':
            base = self.momentum_base
            recent_pnls = self.momentum_pnls
        elif strat_type == 'post_massive':
            base = self.post_massive_base
            recent_pnls = self.post_massive_pnls
        elif strat_type == 'hot_streak':
            base = 0.010  # 1%
            recent_pnls = self.hot_streak_pnls
        elif strat_type == 'hot_streak_3':
            base = 0.008  # smaller for 3-streaks
            recent_pnls = self.hot_streak_pnls
        else:  # hot_streak_5
            base = 0.014  # bit bigger for 5-streaks
            recent_pnls = self.hot_streak_pnls  
        
        mult = 1.0
        
        # give momentum and 5-streaks a boost
        if strat_type == 'momentum':
            mult = 1.5  # momentum is our bread and butter
        elif strat_type == 'hot_streak_5':
            mult = 1.2  # 5 streaks are rare, bet bigger
        
        # adjust based on recent performance
        if len(recent_pnls) >= 10:
            recent_wins = sum(1 for p in recent_pnls if p > 0)
            win_rate = recent_wins / len(recent_pnls)
            
            if strat_type == 'momentum':
                # momentum needs aggressive scaling
                if win_rate > 0.15:  # doing better than expected
                    mult *= 1.5
                elif win_rate > 0.20:  # crushing it
                    mult *= 2.0
                elif win_rate < 0.08:  # ice cold
                    mult *= 0.6
            elif strat_type == 'post_massive':
                # more stable adjustments here
                if win_rate > 0.40:  
                    mult *= 1.3
                elif win_rate < 0.20:  
                    mult *= 0.7
            else:  # hot streaks
                if win_rate > 0.25:  
                    mult *= 1.4
                elif win_rate < 0.15:  
                    mult *= 0.7
        
        # check for win/loss streaks
        if len(recent_pnls) >= 3:
            last_3 = list(recent_pnls)[-3:]
            if all(p > 0 for p in last_3):  # 3 wins straight!
                if strat_type == 'momentum':
                    mult *= 1.3
                elif strat_type in ['hot_streak_3', 'hot_streak', 'hot_streak_5']:
                    mult *= 1.25  
                else:
                    mult *= 1.2
            elif all(p < 0 for p in last_3):  # 3 losses :(
                if strat_type == 'momentum':
                    mult *= 0.6  # cut size hard
                elif strat_type in ['hot_streak_3', 'hot_streak', 'hot_streak_5']:
                    mult *= 0.7  
                else:
                    mult *= 0.8
        
        # drawdown protection - super important!!
        curr_dd = (self.peak_capital - self.capital) / self.peak_capital
        if curr_dd > 0.05:  # more than 5% drawdown
            # gradually reduce size
            dd_mult = 1.0 - curr_dd
            dd_mult = max(dd_mult, 0.5)  # never go below 50%
            mult *= dd_mult
        
        # adjust for account size
        capital_ratio = self.capital / self.init_capital
        if capital_ratio < 0.5:  # lost half :(
            mult *= 0.6
        elif capital_ratio > 3.0:  # tripled up!
            mult *= 1.5
        elif capital_ratio > 2.0:  # doubled
            mult *= 1.3
        
        # momentum special - if last trade was huge, bet more
        if strat_type == 'momentum' and len(self.momentum_pnls) > 0:
            last_pnl = self.momentum_pnls[-1]
            last_return = last_pnl / (self.capital - last_pnl)
            if last_return > 0.20:  # last trade made 20%+
                mult *= 1.5  # PRESS IT
        
        # final bet calculation
        bet = self.capital * base * mult
        
        # max bet limits (dont wanna blow up)
        max_bet = self.capital * 0.05 if strat_type == 'momentum' else self.capital * 0.04
        bet = min(bet, max_bet)
        
        # make sure bet is valid
        return max(bet, 1.0) if bet < self.capital * 0.95 else 0
    
    def exit_check(self, entry_price, curr_price, ticks, target, stop, strat):
        """when to gtfo"""
        ratio = curr_price / entry_price
        
        # hit target or stop
        if curr_price >= target:
            return True, 'target'
        if ratio <= stop:
            return True, 'stop'
        
        # strategy specific exits
        if strat == 'post_massive':
            # tweaked these a lot
            if ratio >= 1.35 and ticks > 60:  # take profit earlier
                return True, 'quick_profit'
            elif ticks > 150:  # timeout
                return True, 'time'
                
        elif strat == 'hot_streak_3':
            # conservative for 3-streaks
            if ratio >= 1.8 and ticks > 80:
                return True, 'warm_profit'
            elif ticks > 120:
                return True, 'time'
                
        elif strat == 'hot_streak':
            # normal hot streak
            if ratio >= 2.5 and ticks > 100:
                return True, 'streak_profit'
            elif ticks > 150:
                return True, 'time'
                
        elif strat == 'hot_streak_5':
            # let 5-streaks run longer
            if ratio >= 4.0 and ticks > 120:
                return True, 'super_streak_profit'
            elif ticks > 200:  
                return True, 'time'
                
        elif strat == 'momentum' and ticks > 300:
            return True, 'time'
        elif ticks > 200:
            return True, 'time'
            
        # moonshot protection - dont be greedy
        if ratio > 10:
            return True, 'moonshot'
            
        return False, None

def backtest(rounds, bot):
    """run the backtest"""
    print("lets gooo... running backtest")
    start = time.time()
    
    sorted_rounds = sorted(rounds.items())
    n_rounds = len(sorted_rounds)
    
    for i, (round_id, data) in enumerate(sorted_rounds[:-1]):
        # track max of each round
        if data is not None and len(data) > 0:
            round_max = max(data)
            bot.last_max = round_max
            
            # add to recent maxes for streak detection
            bot.recent_round_maxes.append(round_max)
            
            # check if was massive win
            if round_max >= 10.0:
                bot.massive_ago = 0
            else:
                bot.massive_ago += 1
        
        # get next round to trade
        next_id, next_data = sorted_rounds[i + 1]
        if next_data is None or len(next_data) <= bot.entry_tick:
            continue
        
        # check entry
        entry_price = next_data[bot.entry_tick]
        strat_type, target, stop = bot.should_trade(entry_price, next_data)
        
        if strat_type is None:
            continue
        
        # calculate bet
        bet = bot.get_bet_size(strat_type)
        if bet <= 0:
            continue
        
        # place trade
        bot.n_trades += 1
        bot.capital -= bet
        
        # update strategy counts
        if strat_type == 'momentum':
            bot.momentum_trades += 1
        elif strat_type == 'post_massive':
            bot.post_massive_trades += 1
        else:  # hot streak
            bot.hot_streak_trades += 1
        
        # simulate the round
        exit_price = 0
        exit_reason = 'crash'  # default to crash
        
        for tick in range(bot.entry_tick + 1, len(next_data)):
            should_exit, reason = bot.exit_check(
                entry_price, next_data[tick], 
                tick - bot.entry_tick, target, stop, strat_type
            )
            
            if should_exit:
                exit_price = next_data[tick]
                exit_reason = reason
                break
        
        # calc pnl
        pnl = bet * (exit_price / entry_price) - bet
        bot.capital += bet + pnl
        
        # update tracking
        bot.recent_pnls.append(pnl)
        if strat_type == 'momentum':
            bot.momentum_pnls.append(pnl)
        elif strat_type == 'post_massive':
            bot.post_massive_pnls.append(pnl)
        else:  
            bot.hot_streak_pnls.append(pnl)
            
        # track peak
        if bot.capital > bot.peak_capital:
            bot.peak_capital = bot.capital
        
        # count wins
        if pnl > 0:
            bot.n_wins += 1
            if strat_type == 'momentum':
                bot.momentum_wins += 1
            elif strat_type == 'post_massive':
                bot.post_massive_wins += 1
            else:  
                bot.hot_streak_wins += 1
        
        # save trade
        bot.trades.append({
            'round': next_id,
            'type': strat_type,
            'entry': entry_price,
            'exit': exit_price,
            'bet': bet,
            'pnl': pnl,
            'capital': bot.capital,
            'reason': exit_reason,
            'return_pct': (exit_price / entry_price - 1) * 100
        })
        
        # progress update
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            curr_ret = (bot.capital - bot.init_capital) / bot.init_capital * 100
            print(f"  processed {i+1}/{n_rounds} rounds... Return: {curr_ret:+.1f}% ({rate:.0f} rounds/sec)")
    
    elapsed = time.time() - start
    print(f"\ndone! took {elapsed:.1f} seconds for {n_rounds} rounds")
    
    return pd.DataFrame(bot.trades)

def show_results(bot, df):
    """show how we did"""
    if len(df) == 0:
        print("No trades! something went wrong...")
        return
    
    returns = (bot.capital - bot.init_capital) / bot.init_capital * 100
    win_rate = bot.n_wins / bot.n_trades * 100
    
    print(f"\n{'='*50}")
    print(f"RESULTS - DID WE BEAT THE HOUSE?")
    print(f"{'='*50}")
    print(f"Total trades: {bot.n_trades}")
    print(f"Winners: {bot.n_wins} ({win_rate:.1f}%)")
    print(f"Rounds skipped: {bot.skipped}")
    print(f"Return: {returns:+.1f}% {'💰' if returns > 0 else '💸'}")
    print(f"Final bankroll: ${bot.capital:,.2f}")
    
    # drawdown check
    peak = bot.peak_capital
    max_dd = (peak - bot.capital) / peak * 100 if peak > bot.capital else 0
    print(f"Peak: ${peak:,.2f}")
    print(f"Current DD: {max_dd:.1f}%")
    
    # breakdown by strat
    print("\nHow each strategy did:")
    
    # momentum stats
    if bot.momentum_trades > 0:
        m_df = df[df['type'] == 'momentum']
        m_wins = bot.momentum_wins
        m_total = m_df['pnl'].sum()
        m_avg = m_total / bot.momentum_trades
        m_wr = m_wins / bot.momentum_trades * 100
        
        print(f"\nMOMENTUM (entry >2x):")
        print(f"  Trades: {bot.momentum_trades}")
        print(f"  Winrate: {m_wr:.1f}%")
        print(f"  Total P&L: ${m_total:,.2f}")
        print(f"  Avg per trade: ${m_avg:.2f}")
        
        # how did we exit
        m_exits = m_df['reason'].value_counts()
        print(f"  Exit breakdown: {dict(m_exits)}")
    
    # post massive stats
    if bot.post_massive_trades > 0:
        p_df = df[df['type'] == 'post_massive']
        p_wins = bot.post_massive_wins
        p_total = p_df['pnl'].sum()
        p_avg = p_total / bot.post_massive_trades
        p_wr = p_wins / bot.post_massive_trades * 100
        
        print(f"\nPOST-MASSIVE COOLING:")
        print(f"  Trades: {bot.post_massive_trades}")
        print(f"  Winrate: {p_wr:.1f}%")
        print(f"  Total P&L: ${p_total:,.2f}")
        print(f"  Avg per trade: ${p_avg:.2f}")
        
        p_exits = p_df['reason'].value_counts()
        print(f"  Exit breakdown: {dict(p_exits)}")
    
    # hot streak stats (all types)
    if bot.hot_streak_trades > 0:
        h_df = df[df['type'].isin(['hot_streak_3', 'hot_streak', 'hot_streak_5'])]
        h_wins = bot.hot_streak_wins
        h_total = h_df['pnl'].sum()
        h_avg = h_total / bot.hot_streak_trades
        h_wr = h_wins / bot.hot_streak_trades * 100
        
        print(f"\nHOT STREAKS (3/4/5 rounds >2x):")
        print(f"  Total trades: {bot.hot_streak_trades}")
        print(f"  Winrate: {h_wr:.1f}%")
        print(f"  Total P&L: ${h_total:,.2f}")
        print(f"  Avg per trade: ${h_avg:.2f}")
        
        # breakdown by streak type
        h3_df = df[df['type'] == 'hot_streak_3']
        h4_df = df[df['type'] == 'hot_streak']
        h5_df = df[df['type'] == 'hot_streak_5']
        
        if len(h3_df) > 0:
            print(f"\n  3-streak (conservative):")
            print(f"    Trades: {len(h3_df)}")
            print(f"    P&L: ${h3_df['pnl'].sum():.2f}")
            
        if len(h4_df) > 0:
            print(f"\n  4-streak (normal):")
            print(f"    Trades: {len(h4_df)}")
            print(f"    P&L: ${h4_df['pnl'].sum():.2f}")
            
        if len(h5_df) > 0:
            print(f"\n  5-streak (aggressive):")
            print(f"    Trades: {len(h5_df)}")
            print(f"    P&L: ${h5_df['pnl'].sum():.2f}")
        
        h_exits = h_df['reason'].value_counts()
        print(f"\n  Exit breakdown: {dict(h_exits)}")
    
    # plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # equity curve
    ax1.plot(df['capital'], 'b-', linewidth=2)
    ax1.axhline(bot.init_capital, color='gray', linestyle='--', alpha=0.5, label='Starting capital')
    ax1.fill_between(range(len(df)), bot.init_capital, df['capital'], 
                     where=df['capital']>bot.init_capital, alpha=0.3, color='green', label='Profit')
    ax1.fill_between(range(len(df)), bot.init_capital, df['capital'], 
                     where=df['capital']<=bot.init_capital, alpha=0.3, color='red', label='Loss')
    ax1.set_title('Account Balance Over Time')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True, alpha=0.3)
    # ax1.legend()  # nah too cluttered
    
    # exit reasons
    exit_counts = df.groupby(['type', 'reason']).size().unstack(fill_value=0)
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow']
    exit_counts.plot(kind='bar', stacked=True, ax=ax2, color=colors[:len(exit_counts.columns)])
    ax2.set_title('How Trades Ended')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Count')
    ax2.legend(title='Exit', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def load_data(folder='total_rounds'):
    """load the crash game data"""
    print(f"loading data from {folder}...")
    start = time.time()
    
    rounds = {}
    base = Path(folder)
    if not base.exists():
        print(f"hmm {folder} doesnt exist, trying default...")
        base = Path('total_rounds')
    
    count = 0
    for subfolder in sorted(base.iterdir()):
        if subfolder.is_dir():
            round_name = subfolder.name
            # look for the csv file
            round_csv = subfolder / f"{round_name}.csv"
            
            if round_csv.exists():
                try:
                    df = pd.read_csv(round_csv)
                    
                    # debug first one
                    if count == 0:
                        print(f"first file: {round_csv.name}")
                        print(f"columns: {list(df.columns)}")
                    
                    if 'multiplier' in df.columns:
                        rounds[round_name] = df['multiplier'].values
                        count += 1
                    # else:
                    #     print(f"no multiplier column in {round_csv}?")
                        
                except Exception as e:
                    print(f"error with {round_csv}: {e}")
            # else:
            #     print(f"cant find {round_csv}")
    
    print(f"loaded {count} rounds in {time.time()-start:.1f}s")
    return rounds


# run it
if __name__ == "__main__":
    # load data
    rounds = load_data()
    if not rounds:
        print("No data! check your folder path")
        exit()
    
    # create bot
    print("\nstarting bot with $1000...")
    bot = ImprovedMomentumBot(capital=1000.0)
    
    # backtest
    trades_df = backtest(rounds, bot)
    
    # show results
    show_results(bot, trades_df)
    
    # save trades
    trades_df.to_csv('my_crash_game_trades.csv', index=False)
    print("\nsaved trades to my_crash_game_trades.csv")