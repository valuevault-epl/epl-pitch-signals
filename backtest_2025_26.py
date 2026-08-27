"""
Historical backtest: walk through the ENTIRE 2025-26 EPL season match by match, in chronological
order, computing matchup signals for each fixture using ONLY data that would genuinely have been
known at that point in time (2024-25 season fully, plus whatever 2025-26 games had already been
played before this one) - i.e. a true walk-forward simulation, no lookahead. Every "Strong"-tier
signal produced this way is graded against what actually happened in that real match, and
counted as a single, independent bet (no parlays, no odds - just win/loss counts).

This reuses the EXACT SAME trend/strength/reliability functions as the live dashboard (imported,
not reimplemented), so there is no risk of the backtest silently drifting from what the live
system actually does - it's a historical replay of the identical model, not a lookalike.
"""
import time
import numpy as np
import pandas as pd

from trend_engine import (
    load_results, build_team_history, league_averages_from_matches, build_all_trends,
)
from matchup_engine import build_fixture_signals

BACKTEST_SEASON = '2526'  # 2025-26
BRIDGE_SEASON = '2425'    # 2024-25 - the only prior season allowed as bridge data


def tier(blended):
    """Identical bands to the dashboard's current JS reliabilityTier(): a direct read on the
    blended hit rate, no sample-size discount."""
    if blended >= 75:
        return 'Strong'
    if blended >= 60:
        return 'Moderate'
    return 'Weak'


def signal_hit_rates(sig):
    """Same team_hit_rate/opponent_hit_rate extraction the dashboard JS does, handling both the
    team_/opponent_ naming (goals/corners/sot) and home_team_/away_team_ naming (match_cards)."""
    team_hit = sig.get('team_hit_rate', sig.get('home_team_hit_rate'))
    opp_hit = sig.get('opponent_hit_rate', sig.get('away_team_hit_rate'))
    team_n = sig.get('team_n', sig.get('home_team_n', 0))
    opp_n = sig.get('opponent_n', sig.get('away_team_n', 0))
    return team_hit, opp_hit, team_n, opp_n


def blended_hit_rate(sig):
    """matchup_engine.py now stores hit rates already flipped to '% support for the call' at
    signal-generation time (the line itself is matchup-specific, so the flip has to happen there,
    not here) - no re-flipping needed, just the average."""
    team_hit, opp_hit, team_n, opp_n = signal_hit_rates(sig)
    if team_hit is None or opp_hit is None:
        return None, None, None
    return (team_hit + opp_hit) / 2, team_n, opp_n


def grade_signal(key, sig, row, home_team, away_team):
    """Return (direction, actual_value, line, won) for one signal against the real match row.
    direction and line are read straight off the signal (both matchup-specific now - see
    matchup_engine.py's recommend_line), not re-derived here."""
    def card_count(is_home):
        y = row['HY'] if is_home else row['AY']
        r = row['HR'] if is_home else row['AR']
        y = 0 if pd.isna(y) else y
        r = 0 if pd.isna(r) else r
        return y + r

    line = sig['line']
    direction = sig['direction']

    if key in ('home_goals', 'away_goals'):
        actual = row['FTHG'] if key == 'home_goals' else row['FTAG']
    elif key in ('home_corners', 'away_corners'):
        actual = row['HC'] if key == 'home_corners' else row['AC']
    elif key in ('home_sot', 'away_sot'):
        actual = row['HST'] if key == 'home_sot' else row['AST']
    elif key == 'match_cards':
        actual = card_count(True) + card_count(False)
    else:
        return None, None, None, None

    won = (actual > line) if direction == 'OVER' else (actual < line)
    return direction, actual, line, won


def run_backtest():
    print("Loading full historical results (already fetched by trend_engine)...")
    results = load_results('E0')
    results_championship = load_results('E1')

    season_matches = results[results['season'] == BACKTEST_SEASON].sort_values('Date').reset_index(drop=True)
    print(f"2025-26 season: {len(season_matches)} matches to walk through, "
          f"{season_matches['Date'].min()} to {season_matches['Date'].max()}")

    all_bets = []
    t0 = time.time()
    for i, row in season_matches.iterrows():
        match_date = row['Date']
        home_team, away_team = row['HomeTeam'], row['AwayTeam']

        # everything strictly BEFORE this match - no lookahead. This naturally includes all of
        # 2024-25 plus whatever 2025-26 games have already been played.
        asof_results = results[results['Date'] < match_date]
        asof_championship = results_championship[results_championship['Date'] < match_date]

        team_games = build_team_history(asof_results, results_fallback=asof_championship,
                                         current_season=BACKTEST_SEASON)
        if home_team not in team_games or away_team not in team_games:
            continue  # first-ever appearance, no history at all yet - skip, nothing to grade

        league_avgs = league_averages_from_matches(asof_results)
        trends = build_all_trends(team_games, league_avgs)
        if home_team not in trends or away_team not in trends:
            continue

        fixture = {'HomeTeam': home_team, 'AwayTeam': away_team}
        signals, _ = build_fixture_signals(fixture, trends)
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
                'date': match_date, 'home': home_team, 'away': away_team,
                'market': key, 'label': sig['label'], 'direction': direction,
                'line': line, 'actual': actual, 'won': won,
                'blended_hit_rate': round(blended, 1), 'n_total': (team_n or 0) + (opp_n or 0),
            })

        if i % 50 == 0:
            print(f"  ...{i}/{len(season_matches)} matches processed, {len(all_bets)} Strong bets so far "
                  f"({time.time()-t0:.0f}s elapsed)")

    print(f"\nDone in {time.time()-t0:.0f}s. Total Strong bets found: {len(all_bets)}")
    return pd.DataFrame(all_bets)


if __name__ == '__main__':
    bets = run_backtest()
    bets.to_csv('backtest_2025_26_results.csv', index=False)

    n = len(bets)
    wins = int(bets['won'].sum())
    losses = n - wins
    win_rate = wins / n * 100 if n else 0
    print(f"\n{'='*60}")
    print(f"2025-26 SEASON BACKTEST - ALL 'STRONG' SIGNALS, AS SINGLES")
    print(f"{'='*60}")
    print(f"Total Strong bets: {n}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.1f}%")

    print(f"\n--- Breakdown by market ---")
    for market in bets['market'].unique():
        sub = bets[bets['market'] == market]
        w = int(sub['won'].sum())
        print(f"  {market}: n={len(sub)}, wins={w}, losses={len(sub)-w}, win_rate={w/len(sub)*100:.1f}%")

    print(f"\n--- Breakdown by direction ---")
    for direction in bets['direction'].unique():
        sub = bets[bets['direction'] == direction]
        w = int(sub['won'].sum())
        print(f"  {direction}: n={len(sub)}, wins={w}, losses={len(sub)-w}, win_rate={w/len(sub)*100:.1f}%")
