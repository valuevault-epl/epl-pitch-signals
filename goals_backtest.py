import pandas as pd

WORKDIR = r"C:\Users\capta\AppData\Local\Temp\claude\C--Users-capta\72b7777a-b095-49a5-8922-0c096ada8217\scratchpad\epl"
df = pd.read_csv(f"{WORKDIR}\\epl_with_goals_model.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['total_goals'] = df['FTHG'] + df['FTAG']
df['over25_win'] = df['total_goals'] > 2.5

seasons = sorted(df['season'].unique())
n = len(seasons)
dev_seasons = seasons[:int(n * 0.6)]
val_seasons = seasons[int(n * 0.6):int(n * 0.8)]
holdout_seasons = seasons[int(n * 0.8):]

print("=== Bet OVER 2.5 only when expected_total_goals is high, by period ===\n")
for thresh in [3.0, 3.25, 3.5]:
    print(f"--- Threshold: exp_total_goals >= {thresh} ---")
    for name, seasons_sub in [('DEV', dev_seasons), ('VALIDATION', val_seasons), ('HOLDOUT', holdout_seasons)]:
        sub = df[df['season'].isin(seasons_sub) & (df['exp_total_goals'] >= thresh) & df['B365>2.5'].notna()]
        if len(sub) < 5:
            print(f"  {name}: n={len(sub)} (too few)")
            continue
        won = sub['over25_win']
        pnl = won * (sub['B365>2.5'] - 1) - (~won) * 1.0
        print(f"  {name}: n={len(sub)}, hit_rate={won.mean()*100:.1f}%, ROI={pnl.sum()/len(sub)*100:.2f}%, total_pnl={pnl.sum():.1f}u")
    print()
