"""
Historical backtest: walk through EVERY available EPL season, match by match in chronological
order within each season, computing matchup signals using ONLY data that would genuinely have
been known at that point in time - i.e. a true walk-forward simulation, no lookahead, repeated
season after season through to whatever of the current season has been played so far.

2000-01 is the earliest season usable for this methodology: football-data.co.uk's corners/
shots-on-target/cards columns don't exist before then (only goals are available pre-2000), so
starting earlier would silently starve 6 of the 8 markets while still nominally "running" a
backtest on them - worse than just being upfront about the boundary. This is also exactly the
range the live dashboard's own trend_engine.py already fetches, so no data-sourcing change was
needed, just generalizing the single-season walk forward into a loop over every season.

Each season uses the SAME adaptive bridge as the live dashboard (see build_team_history in
trend_engine.py): starts from a 10-game window bridged from the immediately preceding season,
dropping the oldest bridged game one at a time as the new season's own games arrive, until the
new season alone has 10 games - after which the bridge stops permanently and the sample just
accumulates (uncapped, up to 38) for the rest of that season. This repeats independently, fresh,
for every season tested - a team's sample resets to the bridge design at the start of each new
season exactly as it would live.

Reuses the EXACT SAME trend/strength/tier functions as the live dashboard and the single-season
backtest (imported, not reimplemented) - no risk of drifting from what the live system does.

Performance note: trends only change when the walk-forward "as of" cutoff date changes, and every
match on the same calendar date shares the same cutoff (the asof filter uses strict '<', so
same-day matches never see each other's results anyway) - so trends are computed once per unique
date within a season, not once per match, and reused across every match that date. Large, safe
speedup (most PL matchdays have several games on the same date) with zero behavior change from
the naive per-match version.
"""
import time
import pandas as pd

from trend_engine import (
    load_results, build_team_history, league_averages_from_matches, build_all_trends,
)
from matchup_engine import build_fixture_signals
from backtest_2025_26 import tier, blended_hit_rate, grade_signal

EARLIEST_SEASON = '0001'  # 2000-01 - first season with corners/SOT/cards columns


def run_full_backtest():
    print("Loading full historical results (already fetched by trend_engine)...")
    results = load_results('E0')
    results_championship = load_results('E1')

    all_seasons = sorted(s for s in results['season'].unique() if s >= EARLIEST_SEASON)
    print(f"Testing {len(all_seasons)} seasons: {all_seasons[0]} to {all_seasons[-1]}")

    all_bets = []
    t0 = time.time()
    for season in all_seasons:
        season_matches = results[results['season'] == season].sort_values('Date').reset_index(drop=True)
        if len(season_matches) == 0:
            continue
        unique_dates = sorted(season_matches['Date'].unique())

        trends_by_date = {}
        for d in unique_dates:
            asof_results = results[results['Date'] < d]
            if len(asof_results) == 0:
                trends_by_date[d] = {}
                continue
            asof_championship = results_championship[results_championship['Date'] < d]
            team_games = build_team_history(asof_results, results_fallback=asof_championship,
                                             current_season=season)
            league_avgs = league_averages_from_matches(asof_results)
            trends_by_date[d] = build_all_trends(team_games, league_avgs)

        season_bets = 0
        for i, row in season_matches.iterrows():
            trends = trends_by_date[row['Date']]
            home_team, away_team = row['HomeTeam'], row['AwayTeam']
            if home_team not in trends or away_team not in trends:
                continue

            fixture = {'HomeTeam': home_team, 'AwayTeam': away_team}
            signals = build_fixture_signals(fixture, trends)
            if not signals:
                continue

            for key, sig in signals.items():
                blended, team_n, opp_n = blended_hit_rate(sig)
                if blended is None:
                    continue
                t = tier(blended)
                if t != 'Strong':
                    continue
                direction, actual, line, won = grade_signal(key, sig, row, home_team, away_team)
                if won is None:
                    continue
                all_bets.append({
                    'season': season, 'date': row['Date'], 'home': home_team, 'away': away_team,
                    'market': key, 'label': sig['label'], 'direction': direction,
                    'line': line, 'actual': actual, 'won': won,
                    'blended_hit_rate': round(blended, 1), 'n_total': (team_n or 0) + (opp_n or 0),
                })
                season_bets += 1

        print(f"  season {season}: {len(season_matches)} matches, {season_bets} Strong bets "
              f"({time.time()-t0:.0f}s elapsed total)", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. Total Strong bets across all seasons: {len(all_bets)}")
    return pd.DataFrame(all_bets)


if __name__ == '__main__':
    bets = run_full_backtest()
    bets.to_csv('backtest_all_seasons_results.csv', index=False)

    n = len(bets)
    wins = int(bets['won'].sum())
    losses = n - wins
    win_rate = wins / n * 100 if n else 0
    print(f"\n{'='*60}")
    print("ALL-SEASONS BACKTEST (2000-01 to present) - ALL 'STRONG' SIGNALS, AS SINGLES")
    print(f"{'='*60}")
    print(f"Total Strong bets: {n}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.1f}%")

    print("\n--- Breakdown by market ---")
    for market in bets['market'].unique():
        sub = bets[bets['market'] == market]
        w = int(sub['won'].sum())
        print(f"  {market}: n={len(sub)}, wins={w}, losses={len(sub)-w}, win_rate={w/len(sub)*100:.1f}%")

    print("\n--- Breakdown by direction ---")
    for direction in bets['direction'].unique():
        sub = bets[bets['direction'] == direction]
        w = int(sub['won'].sum())
        print(f"  {direction}: n={len(sub)}, wins={w}, losses={len(sub)-w}, win_rate={w/len(sub)*100:.1f}%")

    print("\n--- Breakdown by season ---")
    for season in sorted(bets['season'].unique()):
        sub = bets[bets['season'] == season]
        w = int(sub['won'].sum())
        print(f"  {season}: n={len(sub)}, wins={w}, losses={len(sub)-w}, win_rate={w/len(sub)*100:.1f}%")
