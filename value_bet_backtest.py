import numpy as np
import pandas as pd

WORKDIR = r"C:\Users\capta\AppData\Local\Temp\claude\C--Users-capta\72b7777a-b095-49a5-8922-0c096ada8217\scratchpad\epl"
df = pd.read_csv(f"{WORKDIR}\\epl_with_model_probs.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])

seasons = sorted(df['season'].unique())
n = len(seasons)
dev_seasons = seasons[:int(n * 0.6)]
val_seasons = seasons[int(n * 0.6):int(n * 0.8)]
holdout_seasons = seasons[int(n * 0.8):]

for outcome, odds_col, prob_col in [('H', 'B365H', 'pH'), ('D', 'B365D', 'pD'), ('A', 'B365A', 'pA')]:
    df[f'implied_{outcome}'] = 1 / df[odds_col]
overround = df['implied_H'] + df['implied_D'] + df['implied_A']
for outcome in ['H', 'D', 'A']:
    df[f'implied_{outcome}_devigged'] = df[f'implied_{outcome}'] / overround


def backtest(sub, ev_threshold, stake_mode='flat'):
    bets = []
    for _, row in sub.iterrows():
        for outcome, odds_col, prob_col in [('H', 'B365H', 'pH'), ('D', 'B365D', 'pD'), ('A', 'B365A', 'pA')]:
            model_p = row[prob_col]
            odds = row[odds_col]
            ev = model_p * odds - 1
            if ev >= ev_threshold:
                won = (row['FTR'] == outcome)
                if stake_mode == 'flat':
                    stake = 1.0
                else:  # quarter-kelly
                    b = odds - 1
                    kelly_f = (model_p * b - (1 - model_p)) / b if b > 0 else 0
                    stake = max(0, min(kelly_f * 0.25, 0.1))
                pnl = stake * (odds - 1) if won else -stake
                bets.append({'date': row['Date'], 'outcome': outcome, 'odds': odds, 'model_p': model_p,
                             'ev': ev, 'won': won, 'stake': stake, 'pnl': pnl})
    return pd.DataFrame(bets)


print("=== DEV: EV threshold sweep (flat stakes) ===")
dev_sub = df[df['season'].isin(dev_seasons)]
for thresh in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    bets = backtest(dev_sub, thresh)
    if len(bets):
        roi = bets['pnl'].sum() / bets['stake'].sum() * 100
        print(f"  EV>={thresh:.2f}: n_bets={len(bets)}, win_rate={bets['won'].mean()*100:.1f}%, "
              f"total_pnl={bets['pnl'].sum():.1f}u, ROI={roi:.2f}%")
    else:
        print(f"  EV>={thresh:.2f}: no bets")
