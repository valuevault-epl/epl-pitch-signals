import pandas as pd
import glob
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(WORKDIR, "seasons", "E0_*.csv")))

dfs = []
for f in files:
    season = os.path.basename(f).replace("E0_", "").replace(".csv", "")
    try:
        df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip')
    except Exception as e:
        print(f"{season}: read failed - {e}")
        continue
    df['season'] = season
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True, sort=False)
combined['Date'] = pd.to_datetime(combined['Date'], format='mixed', dayfirst=True, errors='coerce')
before = combined.groupby('season').size()
combined = combined.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR']).sort_values('Date').reset_index(drop=True)
after = combined.groupby('season').size()
print("Rows before/after dropna, per season (last 5):")
print(pd.DataFrame({'before': before, 'after': after}).tail(5))

print(f"Total matches: {len(combined)}")
print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
print(f"\nKey odds column availability (non-null %):")
for col in ['B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5', 'AHh', 'B365AHH', 'B365AHA',
            'PSH', 'PSD', 'PSA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA']:
    if col in combined.columns:
        pct = combined[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}%")
    else:
        print(f"  {col}: MISSING")

combined.to_csv(os.path.join(WORKDIR, "epl_combined.csv"), index=False)
print(f"\nSaved to epl_combined.csv")
