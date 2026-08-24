import pandas as pd

df = pd.read_csv(r"C:\Users\capta\AppData\Local\Temp\claude\C--Users-capta\72b7777a-b095-49a5-8922-0c096ada8217\scratchpad\epl\epl_combined.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['B365H', 'B365D', 'B365A', 'FTR']).reset_index(drop=True)

# chronological split by season count (26 seasons total: 2000/01-2025/26)
seasons = sorted(df['season'].unique())
n = len(seasons)
dev_seasons = seasons[:int(n * 0.6)]
val_seasons = seasons[int(n * 0.6):int(n * 0.8)]
holdout_seasons = seasons[int(n * 0.8):]
print(f"DEV: {dev_seasons[0]}-{dev_seasons[-1]} ({len(dev_seasons)} seasons)")
print(f"VALIDATION: {val_seasons[0]}-{val_seasons[-1]} ({len(val_seasons)} seasons)")
print(f"HOLDOUT: {holdout_seasons[0]}-{holdout_seasons[-1]} ({len(holdout_seasons)} seasons)\n")


def stake_result(row, side, stake=1.0):
    odds = {'H': row['B365H'], 'D': row['B365D'], 'A': row['B365A']}[side]
    won = (row['FTR'] == side)
    return stake * (odds - 1) if won else -stake


for name, seasons_subset in [('DEV', dev_seasons), ('VALIDATION', val_seasons), ('HOLDOUT', holdout_seasons)]:
    sub = df[df['season'].isin(seasons_subset)]
    print(f"--- {name} ({len(sub)} matches) ---")
    for label, side_fn in [
        ('Always back HOME', lambda r: 'H'),
        ('Always back AWAY', lambda r: 'A'),
        ('Always back DRAW', lambda r: 'D'),
        ('Always back FAVORITE (lowest odds)', lambda r: min(['H', 'D', 'A'], key=lambda s: r[f'B365{s}'])),
        ('Always back UNDERDOG (highest odds)', lambda r: max(['H', 'D', 'A'], key=lambda s: r[f'B365{s}'])),
    ]:
        pnl = sub.apply(lambda r: stake_result(r, side_fn(r)), axis=1)
        roi = pnl.sum() / len(sub) * 100
        print(f"  {label}: n={len(sub)}, total_pnl={pnl.sum():.1f}u, ROI={roi:.2f}%")
    print()
