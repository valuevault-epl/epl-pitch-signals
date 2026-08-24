"""
Team-form-based expected-total-goals model for the Over/Under 2.5 market - walk-forward (each
team's rolling scoring/conceding average only ever uses matches strictly before the current one).
Calibrated into a probability via a bucket table fit on DEV only, same pattern as elo_model.py.
"""
import numpy as np
import pandas as pd

WORKDIR = r"C:\Users\capta\AppData\Local\Temp\claude\C--Users-capta\72b7777a-b095-49a5-8922-0c096ada8217\scratchpad\epl"


def load_matches():
    df = pd.read_csv(f"{WORKDIR}\\epl_combined.csv", low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['B365H', 'B365D', 'B365A', 'FTR', 'FTHG', 'FTAG']).reset_index(drop=True)
    return df.sort_values('Date').reset_index(drop=True)


def compute_expected_goals(df, window=8):
    """Rolling average goals scored/conceded per team (home and away tracked together)."""
    history = {}  # team -> list of (goals_for, goals_against)
    exp_total = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        h_hist = history.get(h, [])
        a_hist = history.get(a, [])
        if len(h_hist) >= 3 and len(a_hist) >= 3:
            h_recent = h_hist[-window:]
            a_recent = a_hist[-window:]
            h_avg_for = np.mean([x[0] for x in h_recent])
            h_avg_against = np.mean([x[1] for x in h_recent])
            a_avg_for = np.mean([x[0] for x in a_recent])
            a_avg_against = np.mean([x[1] for x in a_recent])
            exp_h_goals = (h_avg_for + a_avg_against) / 2
            exp_a_goals = (a_avg_for + h_avg_against) / 2
            exp_total[i] = exp_h_goals + exp_a_goals

        history.setdefault(h, []).append((row['FTHG'], row['FTAG']))
        history.setdefault(a, []).append((row['FTAG'], row['FTHG']))
    df = df.copy()
    df['exp_total_goals'] = exp_total
    return df


def fit_calibration(df_dev, bucket_width=0.25):
    d = df_dev.dropna(subset=['exp_total_goals']).copy()
    d['bucket'] = (d['exp_total_goals'] // bucket_width) * bucket_width
    d['over25'] = (d['FTHG'] + d['FTAG']) > 2.5
    grouped = d.groupby('bucket')['over25'].agg(['mean', 'count'])
    return grouped, bucket_width


if __name__ == '__main__':
    df = load_matches()
    df = compute_expected_goals(df, window=8)

    seasons = sorted(df['season'].unique())
    n = len(seasons)
    dev_seasons = seasons[:int(n * 0.6)]
    val_seasons = seasons[int(n * 0.6):int(n * 0.8)]
    holdout_seasons = seasons[int(n * 0.8):]

    df_dev = df[df['season'].isin(dev_seasons)]
    calib, bw = fit_calibration(df_dev)
    print("Calibration (P(over 2.5) by expected-total-goals bucket, DEV only):")
    print(calib[calib['count'] >= 10].to_string())

    df.to_csv(f"{WORKDIR}\\epl_with_goals_model.csv", index=False)
    print(f"\nSaved. {df['exp_total_goals'].notna().sum()} matches have a model prediction.")
