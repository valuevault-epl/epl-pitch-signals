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
    compute_team_card_count,
)
from matchup_engine import build_fixture_signals, ANCHOR_LINE
from backtest_2025_26 import tier, blended_hit_rate, grade_signal

MATCH_ARCHIVE_START = '1920'  # 2019-20 - same era boundary as the ledger; recent enough to cover
                               # any realistically-aged tracked bet, small enough to keep the
                               # embedded JSON payload light.

# Bet-slip market keys map directly onto these archive field names (no translation table needed
# on the JS side) - see dashboard_template.html's gradeLeg().
SGM_MARKETS = ['home_goals', 'away_goals', 'home_corners', 'away_corners', 'home_sot', 'away_sot', 'match_cards']

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
            signals, _ = build_fixture_signals({'HomeTeam': home_team, 'AwayTeam': away_team}, trends)
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


def _floor_stats(wins, n, edge):
    win_rate = wins / n if n else 0
    return {
        'win_rate': round(win_rate * 100, 1), 'n': n, 'wins': wins, 'losses': n - wins,
        'min_odds': round((1 + edge) / win_rate, 2) if win_rate > 0 else None,
    }


def _pava_bands(sub, edge, min_band_n=20):
    """Isotonic regression (pool-adjacent-violators) fit DIRECTLY on individual bets sorted by
    predicted (blended) hit rate - not on coarse pre-defined 5-point buckets. Pre-binning first
    and pooling second (the earlier version of this function) forced everything into ~3-4 giant
    buckets whenever adjacent buckets were close, which they usually are - e.g. two different
    signals in the SAME game, both genuinely Strong but at different predicted strengths, would
    show the exact same "take at X odds" because both buckets got pooled together. Fitting PAVA on
    the raw sequence instead gives the finest-grained monotonic step function the data actually
    supports: each bet starts as its own block, and blocks are only merged when a real
    monotonicity violation forces it, so a band is exactly as narrow as the data allows without
    ever showing a stronger predicted line as needing worse odds than a weaker one.

    Plateaus (points that ended up pooled to the same rate) with fewer than `min_band_n` bets are
    merged into a neighboring plateau - individually pooling is still monotonic-safe, but a
    length-1 plateau isn't a trustworthy band on its own. Boundaries between adjacent bands are
    set at the midpoint between their nearest raw values, so every possible blended hit rate in
    [75, 100] falls into exactly one band with no gaps.

    Bets are first grouped by their EXACT predicted value (common - many bets land on the same
    round percentage, e.g. from a 10-game sample) before PAVA runs, so ties are never split across
    two different bands - that would otherwise produce a band with identical low/high (a zero-
    width, mathematically unreachable range), seen directly when this ran on raw rows instead."""
    grouped = sub.groupby('blended_hit_rate')['won'].agg(['sum', 'count']).reset_index()
    grouped = grouped.sort_values('blended_hit_rate')
    predicted = grouped['blended_hit_rate'].tolist()
    group_wins = grouped['sum'].tolist()
    group_n = grouped['count'].tolist()

    stack = []  # each: [wins, n, lo_idx, hi_idx] - lo_idx/hi_idx index into `predicted`/groups
    for i in range(len(predicted)):
        wins, n, lo, hi = int(group_wins[i]), int(group_n[i]), i, i
        while stack and (stack[-1][0] / stack[-1][1]) > (wins / n):
            pwins, pn, plo, phi = stack.pop()
            wins, n, lo = wins + pwins, n + pn, plo
        stack.append([wins, n, lo, hi])

    # Merge any too-thin block into a neighbor - forward for all but the last block (which merges
    # backward instead, having no next block to take). Repeats until every block clears the
    # threshold or only one remains. Merging two ADJACENT already-monotonic blocks can't break
    # monotonicity, so this is safe regardless of merge direction.
    merged = [list(b) for b in stack]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, b in enumerate(merged):
            if b[1] < min_band_n:
                if i < len(merged) - 1:
                    nxt = merged.pop(i + 1)
                    b[0] += nxt[0]
                    b[1] += nxt[1]
                    b[3] = nxt[3]
                else:
                    prev = merged.pop(i - 1)
                    b[0] += prev[0]
                    b[1] += prev[1]
                    b[2] = prev[2]
                changed = True
                break

    bands = []
    for wins, n, lo, hi in merged:
        bands.append({'_raw_low': predicted[lo], '_raw_high': predicted[hi], **_floor_stats(wins, n, edge)})

    for i, b in enumerate(bands):
        b['low'] = 75.0 if i == 0 else round((bands[i - 1]['_raw_high'] + b['_raw_low']) / 2, 1)
        b['high'] = 100.01 if i == len(bands) - 1 else round((b['_raw_high'] + bands[i + 1]['_raw_low']) / 2, 1)
    for b in bands:
        del b['_raw_low'], b['_raw_high']
    return bands


def compute_floors(ledger, edge=0.05):
    """Per-market floor, PLUS per-market bands keyed by the predicted (blended) hit rate at
    grading time, so the dashboard can look up how bets predicted at roughly a given signal's
    current strength have actually performed across thousands of historical bets, instead of
    using a single fixture's own small sample directly as the odds threshold. See _pava_bands for
    how the bands themselves are built (isotonic regression, not fixed-width bucketing) - this is
    what guarantees a stronger predicted line never shows worse odds than a weaker one, while
    staying as fine-grained as the data actually supports."""
    floors = {}
    for market in ledger['market'].unique():
        sub = ledger[ledger['market'] == market]
        floor = _floor_stats(int(sub['won'].sum()), len(sub), edge)
        floor['since_season'] = VAR_ERA_START
        floor['bands'] = _pava_bands(sub, edge)
        floors[market] = floor
    return floors


def build_team_match_archive(results, since_season=MATCH_ARCHIVE_START):
    """Every completed match's actual values for the 7 team markets, keyed by
    "HomeTeam|AwayTeam|YYYY-MM-DD" - what the bet slip's tracker looks a leg's real result up in
    once its fixture has been played. Deliberately separate from each team's own _games log (used
    for the per-team history modal): _games is a ROLLING window that drops old games as new ones
    arrive, so a bet graded weeks after being placed could otherwise fall out of the window it'd
    need to be looked up in. This archive only grows."""
    sub = results[results['season'] >= since_season]
    archive = {}
    for _, row in sub.iterrows():
        home_cards = compute_team_card_count(row, True)
        away_cards = compute_team_card_count(row, False)
        key = f"{row['HomeTeam']}|{row['AwayTeam']}|{row['Date'].strftime('%Y-%m-%d')}"
        archive[key] = {
            'home_goals': None if pd.isna(row['FTHG']) else int(row['FTHG']),
            'away_goals': None if pd.isna(row['FTAG']) else int(row['FTAG']),
            'home_corners': None if pd.isna(row.get('HC')) else int(row['HC']),
            'away_corners': None if pd.isna(row.get('AC')) else int(row['AC']),
            'home_sot': None if pd.isna(row.get('HST')) else int(row['HST']),
            'away_sot': None if pd.isna(row.get('AST')) else int(row['AST']),
            'match_cards': int(home_cards + away_cards),
        }
    return archive


def _actual_values_table(results):
    out = pd.DataFrame(index=results.index)
    out['home_goals'] = results['FTHG']
    out['away_goals'] = results['FTAG']
    out['home_corners'] = results['HC']
    out['away_corners'] = results['AC']
    out['home_sot'] = results['HST']
    out['away_sot'] = results['AST']
    out['match_cards'] = (results.apply(lambda r: compute_team_card_count(r, True), axis=1)
                           + results.apply(lambda r: compute_team_card_count(r, False), axis=1))
    return out


def compute_sgm_lift(results, since_season=MATCH_ARCHIVE_START):
    """Empirical correlation between every pair of team markets, measured directly against real
    match history rather than assumed - this is what lets the slip give a genuine "true SGM odds"
    figure instead of the naive (and known-wrong for same-game legs) independence assumption.

    For each ordered pair of markets and each direction combo, lift = P(both hit) / (P(one hit) x
    P(other hit)). 1.0 means no correlation (independence holds); above 1.0 means the two events
    tend to happen together more than chance would predict (e.g. a high-tempo game producing both
    more goals and more corners); below 1.0 means they pull against each other. Applied
    multiplicatively to the naive joint probability of same-fixture legs in the slip - not a full
    joint distribution (that would need every possible line combination enumerated, intractable),
    but a real, data-grounded adjustment computed at the anchor line and assumed to carry over
    approximately to nearby alt lines, which is disclosed to the user rather than presented as
    exact."""
    sub = results[results['season'] >= since_season]
    actuals = _actual_values_table(sub)
    anchor_map = {
        'home_goals': ANCHOR_LINE['goals_for'], 'away_goals': ANCHOR_LINE['goals_for'],
        'home_corners': ANCHOR_LINE['corners_for'], 'away_corners': ANCHOR_LINE['corners_for'],
        'home_sot': ANCHOR_LINE['shots_on_target_for'], 'away_sot': ANCHOR_LINE['shots_on_target_for'],
        'match_cards': ANCHOR_LINE['match_cards'],
    }
    hits = {}
    for m in SGM_MARKETS:
        anchor = anchor_map[m]
        hits[(m, 'OVER')] = actuals[m] > anchor
        hits[(m, 'UNDER')] = actuals[m] < anchor

    lift = {}
    for m1 in SGM_MARKETS:
        for m2 in SGM_MARKETS:
            if m1 == m2:
                continue
            for d1 in ('OVER', 'UNDER'):
                for d2 in ('OVER', 'UNDER'):
                    a, b = hits[(m1, d1)], hits[(m2, d2)]
                    p_a, p_b = a.mean(), b.mean()
                    p_joint = (a & b).mean()
                    ratio = (p_joint / (p_a * p_b)) if p_a > 0 and p_b > 0 else 1.0
                    lift[f"{m1}|{d1}|{m2}|{d2}"] = round(float(ratio), 3)
    return lift


if __name__ == '__main__':
    ledger = update_ledger()
    floors = compute_floors(ledger)

    print("\n--- Market floors (5% edge minimum odds) ---")
    for market, f in sorted(floors.items(), key=lambda kv: -kv[1]['win_rate']):
        print(f"  {market:16s} win_rate={f['win_rate']:5.1f}%  n={f['n']:5d}  "
              f"min_odds={f['min_odds']}")
        for b in f['bands']:
            print(f"      band {b['low']:>3}-{b['high']:<3}%: actual win_rate={b['win_rate']:5.1f}%  "
                  f"n={b['n']:4d}  min_odds={b['min_odds']}")

    results = load_results('E0')
    match_archive = build_team_match_archive(results)
    sgm_lift = compute_sgm_lift(results)
    print(f"\nBuilt match archive: {len(match_archive)} matches since {MATCH_ARCHIVE_START[:2]}{MATCH_ARCHIVE_START[2:]}")
    print(f"Computed SGM lift for {len(sgm_lift)} market/direction pair combinations")

    trends_path = os.path.join(WORKDIR, 'trends_data.json')
    with open(trends_path) as f:
        data = json.load(f)
    data['market_floors'] = floors
    data['match_archive'] = match_archive
    data['sgm_lift'] = sgm_lift
    with open(trends_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nMerged market_floors, match_archive, sgm_lift into {trends_path}")
