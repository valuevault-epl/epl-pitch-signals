"""
Maintains an ever-growing, never-trimmed ledger of every "Strong"-tier signal since the VAR era
(2019-20 onward) and what actually happened, then derives a per-market minimum-odds floor from
it: the live odds below which a Strong signal in that market wouldn't clear a 5% expected-value
edge, given that market's historical hit rate.

This is DELIBERATELY separate from the adaptive rolling window used for live team trends
(build_team_history in trend_engine.py), which intentionally drops old games so trends reflect
recent form. The floor is the opposite: it should only get more statistically grounded over time,
so var_era_ledger.csv is append-only across runs - each run adds whatever matches have completed
since the ledger's last recorded date and never removes or recomputes older rows. The ledger file
itself is what makes this persist across weekly cloud routine runs (each run starts from a fresh
git clone with no other state), so it must be committed back to the repo every time it changes.

Grading logic (tier, blended_hit_rate, grade_signal) is imported from backtest_2025_26.py rather
than reimplemented, so the floor is always measuring the exact same definition of "Strong" the
live dashboard currently uses - if that definition changes, re-derive the ledger from scratch
(delete var_era_ledger.csv and rerun) rather than let it silently grade two eras by two rules.
"""
import json
import os
import time
import pandas as pd

from trend_engine import (
    load_results, build_team_history, league_averages_from_matches, build_all_trends, WORKDIR,
)
from matchup_engine import build_fixture_signals
from backtest_2025_26 import tier, blended_hit_rate, grade_signal

VAR_ERA_START = '1920'  # 2019-20 - season the ledger starts from
LEDGER_PATH = os.path.join(WORKDIR, 'var_era_ledger.csv')
LEDGER_COLUMNS = ['season', 'date', 'home', 'away', 'market', 'label', 'direction', 'line',
                  'actual', 'won', 'blended_hit_rate', 'n_total']


def _grade_matches(results, results_championship, season_matches):
    """Walk-forward grade every Strong signal for the given matches, date-batched for speed
    (identical logic/results to per-match grading, since same-day matches share an as-of cutoff -
    see backtest_all_seasons.py, which this mirrors)."""
    rows = []
    for season in sorted(season_matches['season'].unique()):
        sm = season_matches[season_matches['season'] == season].sort_values('Date')
        trends_by_date = {}
        for d in sorted(sm['Date'].unique()):
            asof_results = results[results['Date'] < d]
            if len(asof_results) == 0:
                trends_by_date[d] = {}
                continue
            asof_championship = results_championship[results_championship['Date'] < d]
            team_games = build_team_history(asof_results, results_fallback=asof_championship,
                                             current_season=season)
            league_avgs = league_averages_from_matches(asof_results)
            trends_by_date[d] = build_all_trends(team_games, league_avgs)

        for _, row in sm.iterrows():
            trends = trends_by_date[row['Date']]
            home_team, away_team = row['HomeTeam'], row['AwayTeam']
            if home_team not in trends or away_team not in trends:
                continue
            signals = build_fixture_signals({'HomeTeam': home_team, 'AwayTeam': away_team}, trends)
            if not signals:
                continue
            for key, sig in signals.items():
                blended, team_n, opp_n = blended_hit_rate(sig)
                if blended is None or tier(blended) != 'Strong':
                    continue
                direction, actual, line, won = grade_signal(key, sig, row, home_team, away_team)
                if won is None:
                    continue
                rows.append({
                    'season': season, 'date': row['Date'].strftime('%Y-%m-%d'),
                    'home': home_team, 'away': away_team, 'market': key, 'label': sig['label'],
                    'direction': direction, 'line': line, 'actual': actual, 'won': won,
                    'blended_hit_rate': round(blended, 1), 'n_total': (team_n or 0) + (opp_n or 0),
                })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def update_ledger():
    t0 = time.time()
    results = load_results('E0')
    results_championship = load_results('E1')

    if os.path.exists(LEDGER_PATH):
        ledger = pd.read_csv(LEDGER_PATH, dtype={'season': str})
        last_date = pd.to_datetime(ledger['date']).max()
        print(f"Existing ledger: {len(ledger)} bets, latest graded date {last_date.date()}")
        new_matches = results[(results['Date'] > last_date) & (results['season'] >= VAR_ERA_START)]
    else:
        print("No ledger yet - bootstrapping from the full VAR-era history.")
        ledger = pd.DataFrame(columns=LEDGER_COLUMNS)
        new_matches = results[results['season'] >= VAR_ERA_START]

    if len(new_matches) == 0:
        print("No new completed matches since the ledger was last updated.")
    else:
        print(f"Grading {len(new_matches)} newly-completed match(es) not yet in the ledger...")
        new_rows = _grade_matches(results, results_championship, new_matches)
        print(f"  -> {len(new_rows)} new Strong bets added ({time.time()-t0:.0f}s)")
        ledger = pd.concat([ledger, new_rows], ignore_index=True)
        ledger.to_csv(LEDGER_PATH, index=False)

    print(f"Ledger now covers {len(ledger)} bets, {ledger['season'].nunique()} seasons "
          f"({VAR_ERA_START}-present).")
    return ledger


def compute_floors(ledger, edge=0.05):
    floors = {}
    for market in ledger['market'].unique():
        sub = ledger[ledger['market'] == market]
        n = len(sub)
        wins = int(sub['won'].sum())
        win_rate = wins / n if n else 0
        floors[market] = {
            'win_rate': round(win_rate * 100, 1),
            'n': n,
            'wins': wins,
            'losses': n - wins,
            'min_odds': round((1 + edge) / win_rate, 2) if win_rate > 0 else None,
            'since_season': VAR_ERA_START,
        }
    return floors


if __name__ == '__main__':
    ledger = update_ledger()
    floors = compute_floors(ledger)

    print("\n--- Market floors (5% edge minimum odds) ---")
    for market, f in sorted(floors.items(), key=lambda kv: -kv[1]['win_rate']):
        print(f"  {market:16s} win_rate={f['win_rate']:5.1f}%  n={f['n']:5d}  "
              f"min_odds={f['min_odds']}")

    trends_path = os.path.join(WORKDIR, 'trends_data.json')
    with open(trends_path) as f:
        data = json.load(f)
    data['market_floors'] = floors
    with open(trends_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nMerged market_floors into {trends_path}")
