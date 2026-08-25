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
# Rolling window target: 10 recent-form games. Early in a season this bridges from the prior
# season's tail (see build_team_history) so there's always a full 10-game sample; once a team
# has played 10 games in the CURRENT season alone, the bridge stops permanently and the sample
# just keeps growing with every new game for the rest of that season (up to a full 38) - no more
# dropping. So "10" is really "the minimum warm-up size", not a hard cap once the season is under way.
ROLLING_WINDOW = 10
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


def current_season_code(today=None):
    """EPL seasons run Aug-May, coded like '2627' for 2026-27. Derived from the real calendar
    date, not from whichever season the results feed happens to have data for - important right
    at a season's start, when the results file for the new season may not exist yet (zero
    completed matches), which would otherwise make the LAST completed season look like "current"."""
    today = today or datetime.date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


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


def load_fixtures(results=None, current_season=None):
    """football-data.co.uk's fixtures.csv is a separate feed from the season result files, and it
    lags: right after a round is played, the results file already has final scores for those
    matches while fixtures.csv can still list the same already-played matches as 'upcoming' for a
    while. So once we know the current season's completed results, drop any fixture whose
    (HomeTeam, AwayTeam) pairing already has a result there - otherwise the dashboard would show
    finished matches as next week's positions."""
    df = pd.read_csv(os.path.join(WORKDIR, "fixtures_raw.csv"), encoding='utf-8-sig', on_bad_lines='skip')
    e0 = df[df['Div'] == 'E0'].copy()
    e0['Date'] = pd.to_datetime(e0['Date'], format='mixed', dayfirst=True, errors='coerce')
    e0 = e0.dropna(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values('Date').reset_index(drop=True)

    if results is not None and current_season is not None:
        cur = results[results['season'] == current_season]
        played = set(zip(cur['HomeTeam'], cur['AwayTeam']))
        already_played = e0.apply(lambda r: (r['HomeTeam'], r['AwayTeam']) in played, axis=1)
        e0 = e0[~already_played].reset_index(drop=True)

    return e0


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
                        results_fallback=None, min_games=6, current_season=None):
    """For each team, build a list of recent per-game stat dicts. Early in a season, this bridges
    from the PRIOR season's tail (dropping its oldest game one at a time as new current-season
    games arrive) to keep a constant `window`-sized rolling sample - exactly like the previous
    design. But once a team has played `window` games in the CURRENT season alone, the bridge
    stops: prior-season games are dropped entirely and never topped up again, and every new
    current-season game just keeps accumulating (uncapped, up to a full season) rather than
    pushing an old current-season game out. So the sample only ever grows once it's genuinely
    current-season data, and old-season carryover is purely an early-season warm-up device.
    Teams with too little data even after that (newly promoted, fewer than `min_games`) get
    topped up from `results_fallback` (e.g. the Championship), tagged div='Championship' so the
    caveat is visible downstream."""
    seasons_sorted = sorted(results['season'].unique())
    if current_season is None:
        current_season = seasons_sorted[-1]
    prior_seasons = [s for s in seasons_sorted[-use_last_n_seasons:] if s != current_season]

    cur_results = results[results['season'] == current_season].sort_values('Date')
    prior_results = results[results['season'].isin(prior_seasons)].sort_values('Date') if prior_seasons else results.iloc[0:0]

    cur_games = _extract_team_games(cur_results, 'Premier League')
    prior_games = _extract_team_games(prior_results, 'Premier League')

    all_teams = set(cur_games) | set(prior_games)
    team_games = {}
    for team in all_teams:
        cur = cur_games.get(team, [])
        if len(cur) >= window:
            # enough CURRENT-season data alone - use all of it, uncapped, no prior-season bridge
            team_games[team] = cur
        else:
            needed = window - len(cur)
            prior = prior_games.get(team, [])
            team_games[team] = sorted((prior[-needed:] if needed > 0 else []) + cur, key=lambda g: g['date'])

    if results_fallback is not None and len(results_fallback):
        fb_seasons_sorted = sorted(results_fallback['season'].unique())
        fb_recent_seasons = set(fb_seasons_sorted[-use_last_n_seasons:])
        fb_recent = results_fallback[results_fallback['season'].isin(fb_recent_seasons)].sort_values('Date')
        fb_team_games = _extract_team_games(fb_recent, 'Championship')

        for team, fb_games in fb_team_games.items():
            existing = team_games.get(team, [])
            if len(existing) < min_games:  # min_games gates WHETHER to bother, target is still window
                needed = window - len(existing)
                topped_up = (fb_games[-needed:] if needed > 0 else []) + existing
                team_games[team] = sorted(topped_up, key=lambda g: g['date'])

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
    # No [-window:] slice here - build_team_history already decided the right sample (exactly
    # `window` during the early-season bridge, or all of the current season once that's >= window).
    games = team_games.get(team, [])
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

        games = team_games[team]
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
    current_season = current_season_code()
    fixtures = load_fixtures(results, current_season)
    print(f"Loaded {len(fixtures)} upcoming fixtures (already-played matches filtered out)")

    team_games = build_team_history(results, results_fallback=results_championship,
                                     current_season=current_season)
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
