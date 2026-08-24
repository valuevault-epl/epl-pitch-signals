"""
Elo-based team-strength model for EPL match outcomes, walk-forward (ratings only ever use past
matches, never future ones - no lookahead). Probability calibration (Elo-diff -> P(H)/P(D)/P(A))
is fit on DEV data only, then applied unchanged to VALIDATION and HOLDOUT.
"""
import numpy as np
import pandas as pd

WORKDIR = r"C:\Users\capta\AppData\Local\Temp\claude\C--Users-capta\72b7777a-b095-49a5-8922-0c096ada8217\scratchpad\epl"


def load_matches():
    df = pd.read_csv(f"{WORKDIR}\\epl_combined.csv", low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['B365H', 'B365D', 'B365A', 'FTR', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    return df.sort_values('Date').reset_index(drop=True)


def compute_elo(df, k=20, home_adv=60, initial=1500):
    ratings = {}
    elo_diff_at_match = np.zeros(len(df))
    for i, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        rh = ratings.get(h, initial)
        ra = ratings.get(a, initial)
        diff = rh + home_adv - ra
        elo_diff_at_match[i] = diff

        expected_h = 1 / (1 + 10 ** (-diff / 400))
        if row['FTR'] == 'H':
            actual_h = 1.0
        elif row['FTR'] == 'D':
            actual_h = 0.5
        else:
            actual_h = 0.0
        ratings[h] = rh + k * (actual_h - expected_h)
        ratings[a] = ra - k * (actual_h - expected_h)
    df = df.copy()
    df['elo_diff'] = elo_diff_at_match
    return df


def fit_calibration(df_dev, bucket_width=40):
    d = df_dev.copy()
    d['bucket'] = (d['elo_diff'] // bucket_width) * bucket_width
    grouped = d.groupby('bucket')['FTR'].value_counts(normalize=True).unstack(fill_value=0)
    for col in ['H', 'D', 'A']:
        if col not in grouped.columns:
            grouped[col] = 0.0
    return grouped[['H', 'D', 'A']], bucket_width


def apply_calibration(df, calib_table, bucket_width):
    buckets = (df['elo_diff'] // bucket_width) * bucket_width
    known_buckets = calib_table.index.to_numpy()

    def nearest_bucket(b):
        if b in calib_table.index:
            return b
        idx = np.argmin(np.abs(known_buckets - b))
        return known_buckets[idx]

    probs = buckets.apply(nearest_bucket).map(lambda b: calib_table.loc[b])
    prob_df = pd.DataFrame(probs.tolist(), index=df.index, columns=['pH', 'pD', 'pA'])
    return pd.concat([df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)


if __name__ == '__main__':
    df = load_matches()
    df = compute_elo(df, k=20, home_adv=60)

    seasons = sorted(df['season'].unique())
    n = len(seasons)
    dev_seasons = seasons[:int(n * 0.6)]
    val_seasons = seasons[int(n * 0.6):int(n * 0.8)]
    holdout_seasons = seasons[int(n * 0.8):]

    df_dev = df[df['season'].isin(dev_seasons)]
    calib_table, bw = fit_calibration(df_dev)
    print("Calibration table (from DEV only):")
    print(calib_table)

    df_all = apply_calibration(df, calib_table, bw)
    df_all.to_csv(f"{WORKDIR}\\epl_with_model_probs.csv", index=False)
    print(f"\nSaved {len(df_all)} matches with model probabilities.")
