"""
Matchup-based trend engine for EPL teams across multiple markets, in the spirit of a
"gains"-style trends dashboard: for each team, per market, compute a +/- number vs league
average, a hit-rate (% of games crossing a common line), and a sample size / confidence read.
For each upcoming fixture, combine home+away trends into a matchup signal per market.

Markets covered (derived from team-level box score stats, all available in football-data.co.uk):
  - Goals scored / conceded (team total, and match total over/under 2.5)
  - BTTS (both teams to score)
  - Corners won / conceded (team total, and match total over/under 9.5 - common line)
  - Cards (yellow+red) for / against (team total, and match total over/under 3.5 - common line)
  - Shots on target for / against
  - Clean sheets / failed to score
"""
import json
import numpy as np
import pandas as pd
import urllib.request
import datetime
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))
# A full season (38 games) as the base sample, not a short-form window - starts from all of
# 2025-26, and as 2026-27 progresses each new game played adds in while the oldest drops out,
# keeping a constant ~38-game rolling sample per team rather than a small, noisier one.
ROLLING_WINDOW = 38
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def fetch_current_data():
    """Refetch all season files + upcoming fixtures - safe to call repeatedly (weekly)."""
    seasons = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2000, 2027)]
    os.makedirs(os.path.join(WORKDIR, "seasons"), exist_ok=True)
    for div in ['E0', 'E1']:  # E1 = Championship, used as fallback for newly promoted teams
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
            out_path = os.path.join(WORKDIR, "seasons", f"{div}_{season}.csv")
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = r.read()
                if len(data) > 500:
                    with open(out_path, 'wb') as f:
                        f.write(data)
            except Exception:
                pass  # season not started yet, or transient failure - fine, use what we have

    fixtures_path = os.path.join(WORKDIR, "fixtures_raw.csv")
    req = urllib.request.Request("https://www.football-data.co.uk/fixtures.csv", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        with open(fixtures_path, 'wb') as f:
            f.write(r.read())


def load_results(div='E0'):
    import glob
    files = sorted(glob.glob(os.path.join(WORKDIR, "seasons", f"{div}_*.csv")))
    dfs = []
    for f in files:
        season = os.path.basename(f).replace(f"{div}_", "").replace(".csv", "")
        try:
            df = pd.read_csv(f, encoding='latin1', on_bad_lines='skip')
        except Exception:
            continue
        df = df.assign(season=season)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined['Date'] = pd.to_datetime(combined['Date'], format='mixed', dayfirst=True, errors='coerce')
    combined = combined.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR']).sort_values('Date').reset_index(drop=True)
    return combined


def load_fixtures():
    df = pd.read_csv(os.path.join(WORKDIR, "fixtures_raw.csv"), encoding='utf-8-sig', on_bad_lines='skip')
    e0 = df[df['Div'] == 'E0'].copy()
    e0['Date'] = pd.to_datetime(e0['Date'], format='mixed', dayfirst=True, errors='coerce')
    return e0.dropna(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values('Date').reset_index(drop=True)


MARKETS = {
    'goals_for': {'label': 'Goals Scored', 'home_col': 'FTHG', 'away_col': 'FTAG', 'line': 1.5},
    'goals_against': {'label': 'Goals Conceded', 'home_col': 'FTAG', 'away_col': 'FTHG', 'line': 1.5},
    'corners_for': {'label': 'Corners Won', 'home_col': 'HC', 'away_col': 'AC', 'line': 5.5},
    'corners_against': {'label': 'Corners Conceded', 'home_col': 'AC', 'away_col': 'HC', 'line': 5.5},
    'cards_for': {'label': 'Cards Received', 'home_col': None, 'away_col': None, 'line': 1.5},  # computed below
    'shots_on_target_for': {'label': 'Shots on Target', 'home_col': 'HST', 'away_col': 'AST', 'line': 4.5},
    'shots_on_target_against': {'label': 'Shots on Target Conceded', 'home_col': 'AST', 'away_col': 'HST', 'line': 4.5},
}


def compute_team_card_count(row, is_home):
    y = row['HY'] if is_home else row['AY']
    r = row['HR'] if is_home else row['AR']
    y = 0 if pd.isna(y) else y
    r = 0 if pd.isna(r) else r
    return y + r


def _extract_team_games(recent, division_label):
    team_games = {}
    for _, row in recent.iterrows():
        for team, is_home in [(row['HomeTeam'], True), (row['AwayTeam'], False)]:
            stats = {
                'date': row['Date'], 'opponent': row['AwayTeam'] if is_home else row['HomeTeam'],
                'is_home': is_home, 'division': division_label,
                'goals_for': row['FTHG'] if is_home else row['FTAG'],
                'goals_against': row['FTAG'] if is_home else row['FTHG'],
                'corners_for': row.get('HC') if is_home else row.get('AC'),
                'corners_against': row.get('AC') if is_home else row.get('HC'),
                'cards_for': compute_team_card_count(row, is_home),
                'cards_against': compute_team_card_count(row, not is_home),
                'match_total_cards': compute_team_card_count(row, True) + compute_team_card_count(row, False),
                'shots_on_target_for': row.get('HST') if is_home else row.get('AST'),
                'shots_on_target_against': row.get('AST') if is_home else row.get('HST'),
                'btts': (row['FTHG'] > 0) and (row['FTAG'] > 0),
                'clean_sheet': (row['FTAG'] == 0) if is_home else (row['FTHG'] == 0),
            }
            team_games.setdefault(team, []).append(stats)
    return team_games


def build_team_history(results, window=ROLLING_WINDOW, use_last_n_seasons=2,
                        results_fallback=None, min_games=6):
    """For each team, build a list of recent per-game stat dicts (most recent `window`), using
    only the most recent `use_last_n_seasons` seasons (2025-26 + the current season is enough to
    always have a full `window`-sized sample available) so the trend reflects current squad/form,
    not a team's whole 26-year history. Teams with fewer than `min_games` top-flight matches
    (newly promoted) get topped up with their most recent games from `results_fallback` (e.g.
    the Championship), tagged with division='Championship' so the caveat is visible downstream."""
    seasons_sorted = sorted(results['season'].unique())
    recent_seasons = set(seasons_sorted[-use_last_n_seasons:])
    recent = results[results['season'].isin(recent_seasons)].sort_values('Date')
    team_games = _extract_team_games(recent, 'Premier League')

    if results_fallback is not None and len(results_fallback):
        fb_seasons_sorted = sorted(results_fallback['season'].unique())
        fb_recent_seasons = set(fb_seasons_sorted[-use_last_n_seasons:])
        fb_recent = results_fallback[results_fallback['season'].isin(fb_recent_seasons)].sort_values('Date')
        fb_team_games = _extract_team_games(fb_recent, 'Championship')

        for team, fb_games in fb_team_games.items():
            existing = team_games.get(team, [])
            if len(existing) < min_games:
                needed = window - len(existing)
                topped_up = (fb_games[-needed:] if needed > 0 else []) + existing
                team_games[team] = sorted(topped_up, key=lambda g: g['date'])

    for team in team_games:
        team_games[team] = team_games[team][-window * 3:]  # keep a bit extra, we'll slice per-metric
    return team_games


def league_averages_from_matches(results, use_last_n_seasons=2):
    """Compute league-average per-team-per-game stats directly from the match table (not from
    per-team windows), so goals_for/goals_against etc. are exactly symmetric by construction -
    every match contributes one team's 'for' value and the opponent's matching 'against' value."""
    seasons_sorted = sorted(results['season'].unique())
    recent_seasons = set(seasons_sorted[-use_last_n_seasons:])
    recent = results[results['season'].isin(recent_seasons)]
    n = len(recent) * 2  # each match = 2 team-game observations

    def card_total(df, home):
        y = df['HY'] if home else df['AY']
        r = df['HR'] if home else df['AR']
        return y.fillna(0) + r.fillna(0)

    sums = {
        'goals_for': recent['FTHG'].sum() + recent['FTAG'].sum(),
        'goals_against': recent['FTAG'].sum() + recent['FTHG'].sum(),
        'corners_for': recent['HC'].sum() + recent['AC'].sum(),
        'corners_against': recent['AC'].sum() + recent['HC'].sum(),
        'cards_for': card_total(recent, True).sum() + card_total(recent, False).sum(),
        'match_total_cards': (card_total(recent, True) + card_total(recent, False)).sum() * 2,
        'shots_on_target_for': recent['HST'].sum() + recent['AST'].sum(),
        'shots_on_target_against': recent['AST'].sum() + recent['HST'].sum(),
    }
    return {k: v / n for k, v in sums.items()}


def team_trend(team_games, team, stat, window, league_avg, line):
    games = team_games.get(team, [])[-window:]
    vals = [g[stat] for g in games if g.get(stat) is not None and not (isinstance(g[stat], float) and np.isnan(g[stat]))]
    if len(vals) < 4:
        return None
    avg = np.mean(vals)
    hit_rate = np.mean([v > line for v in vals]) * 100
    plus_minus = avg - league_avg
    return {'n': len(vals), 'avg': round(float(avg), 2), 'plus_minus': round(float(plus_minus), 2),
            'hit_rate': round(float(hit_rate), 1), 'league_avg': round(float(league_avg), 2), 'line': line}


def build_all_trends(team_games, league_avgs, window=ROLLING_WINDOW):
    metrics = ['goals_for', 'goals_against', 'corners_for', 'corners_against',
               'cards_for', 'match_total_cards', 'shots_on_target_for', 'shots_on_target_against']
    lines = {'goals_for': 1.5, 'goals_against': 1.5, 'corners_for': 5.5, 'corners_against': 5.5,
             'cards_for': 1.5, 'match_total_cards': 3.5, 'shots_on_target_for': 4.5, 'shots_on_target_against': 4.5}

    trends = {}
    for team in team_games:
        trends[team] = {}
        for m in metrics:
            t = team_trend(team_games, team, m, window, league_avgs[m], lines[m])
            if t:
                trends[team][m] = t

        games = team_games[team][-window:]
        if len(games) >= 4:
            btts_vals = [g['btts'] for g in games]
            trends[team]['btts'] = {'n': len(btts_vals), 'hit_rate': round(float(np.mean(btts_vals)) * 100, 1)}
            cs_vals = [g['clean_sheet'] for g in games]
            trends[team]['clean_sheet'] = {'n': len(cs_vals), 'hit_rate': round(float(np.mean(cs_vals)) * 100, 1)}

        n_champ = sum(1 for g in games if g.get('division') == 'Championship')
        if n_champ > 0:
            trends[team]['_data_note'] = f"Includes {n_champ} Championship game(s) - newly promoted team, limited top-flight sample"
    return trends


if __name__ == '__main__':
    print("Fetching latest data...")
    fetch_current_data()
    results = load_results('E0')
    results_championship = load_results('E1')
    print(f"Loaded {len(results)} PL results (latest: {results['Date'].max()}), "
          f"{len(results_championship)} Championship results (fallback for promoted teams)")
    fixtures = load_fixtures()
    print(f"Loaded {len(fixtures)} upcoming fixtures")

    team_games = build_team_history(results, results_fallback=results_championship)
    league_avgs = league_averages_from_matches(results)
    trends = build_all_trends(team_games, league_avgs)
    print(f"\nComputed trends for {len(trends)} teams")
    print(f"League averages: {league_avgs}")

    out = {'generated_at': datetime.datetime.now().isoformat(), 'league_avgs': league_avgs,
           'trends': trends, 'fixtures': fixtures[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']].assign(
               Date=lambda d: d['Date'].dt.strftime('%Y-%m-%d')).to_dict('records')}
    with open(os.path.join(WORKDIR, 'trends_data.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print("Saved trends_data.json")
