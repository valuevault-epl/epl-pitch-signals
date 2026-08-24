import pandas as pd

WORKDIR = r"C:\Users\capta\AppData\Local\Temp\claude\C--Users-capta\72b7777a-b095-49a5-8922-0c096ada8217\scratchpad\epl"
df = pd.read_csv(f"{WORKDIR}\\epl_combined.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['B365H', 'B365D', 'B365A', 'FTR']).reset_index(drop=True)

seasons = sorted(df['season'].unique())
n = len(seasons)
dev_seasons = seasons[:int(n * 0.6)]
val_seasons = seasons[int(n * 0.6):int(n * 0.8)]
holdout_seasons = seasons[int(n * 0.8):]

# Double chance odds derived from 1X2 odds (fair-ish, ignoring the extra margin a book would
# actually apply to a real double-chance market - this is an approximation using 1/(p1+p2))
df['imp_H'] = 1 / df['B365H']
df['imp_D'] = 1 / df['B365D']
df['imp_A'] = 1 / df['B365A']
df['dc_1X_odds'] = 1 / (df['imp_H'] + df['imp_D'])   # Home or Draw
df['dc_X2_odds'] = 1 / (df['imp_D'] + df['imp_A'])   # Draw or Away
df['dc_12_odds'] = 1 / (df['imp_H'] + df['imp_A'])   # Home or Away (no draw)

df['dc_1X_win'] = df['FTR'].isin(['H', 'D'])
df['dc_X2_win'] = df['FTR'].isin(['D', 'A'])
df['dc_12_win'] = df['FTR'].isin(['H', 'A'])

# Over/Under 2.5 goals (where available)
has_ou = df['B365>2.5'].notna()
df['total_goals'] = df['FTHG'] + df['FTAG']
df['over25_win'] = df['total_goals'] > 2.5
df['under25_win'] = df['total_goals'] <= 2.5


def eval_strategy(sub, win_col, odds_col, filt=None):
    s = sub if filt is None else sub[filt]
    if len(s) == 0:
        return None
    won = s[win_col]
    pnl = won * (s[odds_col] - 1) - (~won) * 1.0
    return {'n': len(s), 'hit_rate': won.mean() * 100, 'roi': pnl.sum() / len(s) * 100, 'total_pnl': pnl.sum()}


print("=== Double Chance strategies (derived odds), by period ===\n")
for name, seasons_sub in [('DEV', dev_seasons), ('VALIDATION', val_seasons), ('HOLDOUT', holdout_seasons)]:
    sub = df[df['season'].isin(seasons_sub)]
    print(f"--- {name} ---")
    for label, win_col, odds_col in [
        ('Home-or-Draw (1X)', 'dc_1X_win', 'dc_1X_odds'),
        ('Draw-or-Away (X2)', 'dc_X2_win', 'dc_X2_odds'),
        ('Home-or-Away (12)', 'dc_12_win', 'dc_12_odds'),
    ]:
        r = eval_strategy(sub, win_col, odds_col)
        print(f"  {label}: n={r['n']}, hit_rate={r['hit_rate']:.1f}%, ROI={r['roi']:.2f}%, total_pnl={r['total_pnl']:.1f}u")

    # Home-or-Draw specifically when home team is favorite (odds < away odds) - "safe favorite" bet
    fav_home = sub['B365H'] < sub['B365A']
    r = eval_strategy(sub, 'dc_1X_win', 'dc_1X_odds', filt=fav_home)
    print(f"  Home-or-Draw WHEN HOME FAVORITE: n={r['n']}, hit_rate={r['hit_rate']:.1f}%, ROI={r['roi']:.2f}%, total_pnl={r['total_pnl']:.1f}u")
    print()

print("\n=== Over/Under 2.5 goals, by period (only seasons with this market) ===\n")
for name, seasons_sub in [('DEV', dev_seasons), ('VALIDATION', val_seasons), ('HOLDOUT', holdout_seasons)]:
    sub = df[df['season'].isin(seasons_sub) & has_ou]
    if len(sub) == 0:
        print(f"--- {name}: no O/U data ---")
        continue
    print(f"--- {name} ({len(sub)} matches with O/U odds) ---")
    r_over = eval_strategy(sub, 'over25_win', 'B365>2.5')
    r_under = eval_strategy(sub, 'under25_win', 'B365<2.5')
    print(f"  Always OVER 2.5: n={r_over['n']}, hit_rate={r_over['hit_rate']:.1f}%, ROI={r_over['roi']:.2f}%")
    print(f"  Always UNDER 2.5: n={r_under['n']}, hit_rate={r_under['hit_rate']:.1f}%, ROI={r_under['roi']:.2f}%")
    print()
