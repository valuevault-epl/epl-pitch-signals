"""
Matchup-based trend engine for EPL teams across multiple markets, in the spirit of a
"gains"-style trends dashboard: for each team, per market, compute a +/- number vs league
average, a hit-rate (% of games crossing a common line), and a sample size / confidence read.
For each upcoming fixture, combine home+away trends into a matchup signal per market.

Markets covered (derived from team-level box score stats, all available in football-data.co.uk):
  - Goals scored / conceded
  - Corners won / conceded
  - Cards (yellow+red) for / against (team total, and match total)
  - Shots on target for / against
  - Clean sheets / failed to score

Lines are not fixed constants (5.5 corners, 1.5 goals, etc.) - bookmakers offer alt lines for
these markets, so instead of testing one hardcoded line, each signal recommends its own line
derived from the projection with a bit of leeway built in (see LEEWAY / recommend_line in
matchup_engine.py), and the hit-rate shown is each team's own history AT that recommended line.
"""
import glob
import json
import re
import time
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

# football-data.co.uk's fixtures.csv (upcoming matches) is a separate feed from its season result
# files and can lag for days after a round finishes before it's updated with the next round's
# dates - confirmed by checking its Last-Modified header, which sat on the previous round's date
# well after that round had been played in full. openfootball's community-maintained season file
# has the full schedule (all 380 fixture dates, known well ahead of kickoff) and updates promptly,
# so it's used as the PRIMARY source for "what's the next round and when" - football-data.co.uk
# remains the only source for results/trends (box-score stats openfootball doesn't have) and is
# kept as a fallback fixture source if openfootball can't be reached.
OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/england/master/{season}/1-premierleague.txt"
OPENFOOTBALL_MONTHS = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
# openfootball spells out full club names; map to football-data.co.uk's short names so team
# lookups against the trends dict (keyed by the football-data.co.uk names) work directly.
OPENFOOTBALL_TEAM_MAP = {
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry", "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton",
    "Fulham FC": "Fulham", "Hull City AFC": "Hull", "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds", "Liverpool FC": "Liverpool", "Manchester City FC": "Man City",
    "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest", "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham", "Burnley FC": "Burnley", "Leicester City FC": "Leicester",
    "Southampton FC": "Southampton", "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves", "Norwich City FC": "Norwich", "Watford FC": "Watford",
    "West Bromwich Albion FC": "West Brom", "Sheffield United FC": "Sheffield United",
    "Luton Town FC": "Luton",
}


def _fetch_openfootball_matchdays(season_start_year):
    """Parse openfootball's plain-text season fixture list into an ordered list of matchdays, each
    a list of (date, HomeTeam, AwayTeam) tuples - position in the list IS the round number
    (index 0 = Matchday 1, matching the "Matchday N" markers in the source file). Shared by
    everything below that needs the full season schedule rather than just the next unplayed round.
    Returns [] on any fetch/parse problem so callers can fall back to another source - this is a
    convenience supplement, not something that should ever crash the pipeline."""
    season = f"{season_start_year}-{(season_start_year + 1) % 100:02d}"
    url = OPENFOOTBALL_URL.format(season=season)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode('utf-8')
    except Exception:
        return []

    day_re = re.compile(
        r'^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+(\d{1,2})(?:\s+(\d{4}))?\s*$')
    match_re = re.compile(r'^\s*(?:\d{1,2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)\s*$')
    score_re = re.compile(r'^(.*?)\s{2,}\d+-\d+(?:\s*\(\d+-\d+\))?\s*$')

    current_year = season_start_year
    current_date = None
    matchdays = []
    cur = None
    try:
        for line in text.splitlines():
            if re.search(r'Matchday\s+\d+', line):
                if cur is not None:
                    matchdays.append(cur)
                cur = []
                continue
            if cur is None:
                continue
            dm = day_re.match(line)
            if dm:
                mon, day, yr = dm.groups()
                if yr:
                    current_year = int(yr)
                current_date = datetime.date(current_year, OPENFOOTBALL_MONTHS[mon], int(day))
                continue
            mm = match_re.match(line)
            if mm and current_date is not None and ' v ' in line:
                home_raw, away_raw = mm.groups()
                score_m = score_re.match(away_raw)
                away_raw = score_m.group(1) if score_m else away_raw
                home = OPENFOOTBALL_TEAM_MAP.get(home_raw.strip(), home_raw.strip())
                away = OPENFOOTBALL_TEAM_MAP.get(away_raw.strip(), away_raw.strip())
                cur.append((current_date, home, away))
        if cur is not None:
            matchdays.append(cur)
    except Exception:
        return []
    return matchdays


def fetch_openfootball_next_round(season_start_year, played_pairs):
    """The next round that hasn't been fully played yet, as
    [{'Date': 'YYYY-MM-DD', 'HomeTeam': ..., 'AwayTeam': ...}, ...]. `played_pairs` is the set of
    (HomeTeam, AwayTeam) already confirmed played, from football-data.co.uk's own results -
    openfootball's OWN score column isn't trusted for this, because it can itself lag on filling in
    final scores for the last few matches of a round even after football-data.co.uk already has
    them (confirmed: it once showed 4 of matchday 1's 10 games as still scoreless a full day after
    they'd finished with recorded results elsewhere). Returns [] if openfootball can't be reached."""
    matchdays = _fetch_openfootball_matchdays(season_start_year)
    for md in matchdays:
        unplayed = [(d, h, a) for (d, h, a) in md if (h, a) not in played_pairs]
        if unplayed:
            return [{'Date': d.isoformat(), 'HomeTeam': h, 'AwayTeam': a} for (d, h, a) in unplayed]
    return []


def fetch_openfootball_rounds(season_start_year, played_pairs):
    """Every fixture of the season mapped to its round number, as {"HomeTeam|AwayTeam": round_num}
    (1-indexed) - what the tracker uses to group a bet's legs by round. Also returns which round
    counts as "current": the first round with any fixture not yet in `played_pairs`, the same
    definition fetch_openfootball_next_round uses for "what's next" - a round that's mid-way
    through being played (some results in, some not) still counts as current rather than jumping
    ahead to the following one. Falls back to the last parsed round once the whole season is
    complete. Returns ({}, None) if openfootball can't be reached."""
    matchdays = _fetch_openfootball_matchdays(season_start_year)
    if not matchdays:
        return {}, None
    match_rounds = {}
    for round_num, md in enumerate(matchdays, start=1):
        for (_, h, a) in md:
            match_rounds[f"{h}|{a}"] = round_num
    current_round = None
    for round_num, md in enumerate(matchdays, start=1):
        if any((h, a) not in played_pairs for (_, h, a) in md):
            current_round = round_num
            break
    if current_round is None:
        current_round = len(matchdays)
    return match_rounds, current_round


# football-data.co.uk's season results CSV stays the sole source for every historical trend/floor
# computation and the var_era_ledger - mixing stat-counting conventions between providers (e.g.
# what counts as a "shot on target") into that permanent record would be a much worse problem than
# a slow refresh. But football-data.co.uk can lag a day or more posting a round's results (or, on a
# bad day, be down entirely - see fetch_current_data's cache fallback above), which is exactly the
# window that leaves "what's been played" stale right when it matters most: a just-finished bet
# that needs grading, or a round that should have already advanced. ESPN's free (undocumented, no
# official ToS coverage - same risk profile as understat.com, which this project already scrapes)
# scoreboard is typically same-day, so it's used ONLY to keep that "played" signal current - never
# merged into `results` itself, and never allowed to override a football-data.co.uk entry that
# already exists for the same match. market_floor.py's fetch_espn_recent_matches does the same job
# with full box-score detail for match_archive grading; this lighter version only needs to know
# WHICH pairs are done, not by how much, so it's a single scoreboard call rather than one call per
# match.
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"

# ESPN spells out full club names; map to football-data.co.uk's short names (verified directly
# against ESPN's own /teams endpoint for the current 20 PL clubs) so pairs line up with what
# results/fixtures/tracked bets already use. A team ESPN returns that isn't in this map (mid-season
# promotion/relegation naming drift) is skipped rather than guessed at.
ESPN_TEAM_MAP = {
    'AFC Bournemouth': 'Bournemouth', 'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa',
    'Brentford': 'Brentford', 'Brighton & Hove Albion': 'Brighton', 'Chelsea': 'Chelsea',
    'Coventry City': 'Coventry', 'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton',
    'Fulham': 'Fulham', 'Hull City': 'Hull', 'Ipswich Town': 'Ipswich', 'Leeds United': 'Leeds',
    'Liverpool': 'Liverpool', 'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest", 'Sunderland': 'Sunderland',
    'Tottenham Hotspur': 'Tottenham',
}


# ESPN's own headers, not the shared HEADERS every other fetch in this file uses - confirmed via
# an actual GitHub Actions run that the bare 'Mozilla/5.0' User-Agent alone gets a flat 403 from
# ESPN specifically when the request comes from a GitHub-hosted runner (works fine from a normal
# residential/dev connection - likely IP-reputation-based bot filtering on ESPN's side, common for
# an API with no official public access). A fuller, more genuinely-browser-shaped header set is
# the standard way past that kind of filtering; scoped to ESPN alone since football-data.co.uk and
# openfootball have shown no sign of minding the plain HEADERS they've always gotten.
ESPN_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.espn.com/',
    'Origin': 'https://www.espn.com',
}


def _espn_get(url):
    req = urllib.request.Request(url, headers=ESPN_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode('utf-8'))


def fetch_espn_recent_played_pairs(days_back=6):
    """(HomeTeam, AwayTeam) pairs ESPN shows as completed in the last `days_back` days - a single,
    cheap scoreboard call (unlike market_floor.fetch_espn_recent_matches, this never fetches a
    per-match box score) used only to keep played/not-played current when football-data.co.uk's
    own results are stale or unreachable. Returns an empty set on any fetch/parse problem, same
    convention as fetch_openfootball_next_round: a convenience supplement that should never block
    the pipeline it's supplementing."""
    today = datetime.date.today()
    start = today - datetime.timedelta(days=days_back)
    date_range = f"{start.strftime('%Y%m%d')}-{today.strftime('%Y%m%d')}"
    try:
        board = _espn_get(f"{ESPN_SCOREBOARD_URL}?dates={date_range}")
    except Exception as e:
        print(f"    fetch_espn_recent_played_pairs failed ({type(e).__name__}: {e})")
        return set()
    pairs = set()
    for event in board.get('events', []):
        try:
            comp = event['competitions'][0]
            if not comp['status']['type'].get('completed'):
                continue
            home_c = next(t for t in comp['competitors'] if t['homeAway'] == 'home')
            away_c = next(t for t in comp['competitors'] if t['homeAway'] == 'away')
            home = ESPN_TEAM_MAP.get(home_c['team']['displayName'])
            away = ESPN_TEAM_MAP.get(away_c['team']['displayName'])
            if home and away:
                pairs.add((home, away))
        except Exception:
            continue  # one malformed event shouldn't drop every other one
    return pairs


# Box-score detail for grading (market_floor.fetch_espn_recent_matches) - shared here so there's
# one place that knows how to read an ESPN event's box score.
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary"


def _espn_match_boxscore(event):
    """One completed ESPN scoreboard event's box score, or None if it isn't usable (not completed
    yet, a team not in ESPN_TEAM_MAP, or a fetch/parse problem - never allowed to raise and take
    down the rest of the season's fetch with it). Yellow/red cards are kept separate (not summed)
    since football-data.co.uk's own HY/AY/HR/AR columns - and everything downstream that reads
    them - expect them that way."""
    try:
        comp = event['competitions'][0]
        if not comp['status']['type'].get('completed'):
            return None
        home_c = next(t for t in comp['competitors'] if t['homeAway'] == 'home')
        away_c = next(t for t in comp['competitors'] if t['homeAway'] == 'away')
        home = ESPN_TEAM_MAP.get(home_c['team']['displayName'])
        away = ESPN_TEAM_MAP.get(away_c['team']['displayName'])
        if not home or not away:
            return None
        match_date = event['date'][:10]  # ISO date prefix, e.g. "2026-08-30T13:00Z" -> date

        summary = _espn_get(f"{ESPN_SUMMARY_URL}?event={event['id']}")
        time.sleep(0.3)  # polite pacing against an undocumented, unrate-limited-by-us endpoint
        stat_teams = {t['team']['displayName']: t.get('statistics', [])
                      for t in summary.get('boxscore', {}).get('teams', [])}

        def stat(team_name, stat_name):
            for s in stat_teams.get(team_name, []):
                if s.get('name') == stat_name:
                    try:
                        return int(float(s['displayValue']))
                    except (TypeError, ValueError):
                        return None
            return None

        home_name, away_name = home_c['team']['displayName'], away_c['team']['displayName']
        return {
            'home': home, 'away': away, 'date': match_date,
            'home_goals': int(home_c['score']), 'away_goals': int(away_c['score']),
            'home_corners': stat(home_name, 'wonCorners'), 'away_corners': stat(away_name, 'wonCorners'),
            'home_sot': stat(home_name, 'shotsOnTarget'), 'away_sot': stat(away_name, 'shotsOnTarget'),
            'home_yellow': stat(home_name, 'yellowCards') or 0, 'away_yellow': stat(away_name, 'yellowCards') or 0,
            'home_red': stat(home_name, 'redCards') or 0, 'away_red': stat(away_name, 'redCards') or 0,
        }
    except Exception:
        return None


def _fetch_with_retries(url, retries=2, backoff=1.5):
    """A handful of the 54 back-to-back requests fetch_current_data fires at football-data.co.uk
    (27 seasons x 2 divisions, no delay between them) failing with a transient 503 is one thing -
    that's what the per-file try/except below already tolerates. ALL of them failing at once
    (confirmed in production: a GitHub Actions run got zero usable season files, crashing
    load_results downstream with a cryptic "No objects to concatenate" rather than anything
    pointing at the real cause) looks more like momentary rate-limiting from firing that many
    requests at one host with no pacing at all, which a short retry-with-backoff can ride out
    without needing to slow down the happy-path case where nothing is wrong."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise last_err


def fetch_current_data():
    """Refetch all season files + upcoming fixtures - safe to call repeatedly (weekly)."""
    seasons = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2000, 2027)]
    os.makedirs(os.path.join(WORKDIR, "seasons"), exist_ok=True)
    ok, failed = 0, 0
    for div in ['E0', 'E1']:  # E1 = Championship, used as fallback for newly promoted teams
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
            out_path = os.path.join(WORKDIR, "seasons", f"{div}_{season}.csv")
            try:
                data = _fetch_with_retries(url)
                if len(data) > 500:
                    with open(out_path, 'wb') as f:
                        f.write(data)
                    ok += 1
            except Exception:
                failed += 1  # season not started yet, or a fetch that never recovered - fine
                             # individually, use what we have; see the check just below for when
                             # EVERY file failed, which isn't fine at all.

    print(f"  Season file fetch: {ok} succeeded, {failed} failed/skipped "
          f"(unstarted seasons count as failed here too, so some are always expected).")
    if ok == 0:
        # A total outage used to always be fatal here, on the reasoning that letting
        # load_results() crash later with a cryptic pandas error was worse. But that meant a
        # website hiccup blocked the ENTIRE pipeline every time - including trends, the odds
        # floor, and match_archive grading, none of which actually need a fresh fetch on any run
        # where yesterday's (or last week's) season files are already sitting on disk from the
        # last successful one. Season results only ever grow, never change retroactively, so a
        # stale file is still completely correct, just missing the very latest match(es) - exactly
        # the gap fetch_espn_recent_matches (market_floor.py) and fetch_espn_recent_played_pairs
        # below exist to paper over. Only truly unrecoverable when there's NOTHING cached at all
        # (a fresh checkout with zero prior successful runs) - genuinely nothing to build from.
        existing = glob.glob(os.path.join(WORKDIR, "seasons", "*.csv"))
        if not existing:
            raise RuntimeError(
                "Every single football-data.co.uk season-file fetch failed this run, and there's "
                "no previously-cached season file on disk to fall back to - nothing to build "
                "trends from at all. Likely a site outage or rate-limiting; try again shortly.")
        print(f"  All fetches failed, but {len(existing)} cached season file(s) from a previous "
              f"run are still on disk - continuing with that (slightly stale) data instead of "
              f"blocking the whole pipeline. ESPN fills the most recent gap for grading.")

    # Same tolerance as the season loop above, and for the same reason: confirmed in production
    # that football-data.co.uk can return a transient 503 here, and this used to be unguarded -
    # one flaky request crashed trend_engine.py before match_archive (built later, from the
    # season files already safely fetched above) ever got a chance to update, silently stalling
    # every tracked bet's grading for as long as the outage lasted. load_fixtures() below falls
    # back to whatever fixtures_raw.csv is already on disk (or an empty fixture list, on a fresh
    # checkout with none yet) if this fetch fails, so the rest of the pipeline - trends, the odds
    # floor, and crucially match_archive - still completes and the tracker keeps grading.
    fixtures_path = os.path.join(WORKDIR, "fixtures_raw.csv")
    try:
        data = _fetch_with_retries("https://www.football-data.co.uk/fixtures.csv")
        if len(data) > 500:
            with open(fixtures_path, 'wb') as f:
                f.write(data)
    except Exception as e:
        print(f"  Could not fetch fixtures.csv ({e}) - will use whatever's already on disk, if anything.")


def current_season_code(today=None):
    """EPL seasons run Aug-May, coded like '2627' for 2026-27. Derived from the real calendar
    date, not from whichever season the results feed happens to have data for - important right
    at a season's start, when the results file for the new season may not exist yet (zero
    completed matches), which would otherwise make the LAST completed season look like "current"."""
    today = today or datetime.date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def load_results(div='E0'):
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


def played_pairs_for_season(results, current_season):
    """(HomeTeam, AwayTeam) pairings with a confirmed result in the current season, per
    football-data.co.uk's own results file - the authoritative played/not-played signal, used to
    filter stale fixture feeds regardless of which source they came from."""
    cur = results[results['season'] == current_season]
    return set(zip(cur['HomeTeam'], cur['AwayTeam']))


def load_fixtures(results=None, current_season=None):
    """football-data.co.uk's fixtures.csv is a separate feed from the season result files, and it
    lags: right after a round is played, the results file already has final scores for those
    matches while fixtures.csv can still list the same already-played matches as 'upcoming' for a
    while. So once we know the current season's completed results, drop any fixture whose
    (HomeTeam, AwayTeam) pairing already has a result there - otherwise the dashboard would show
    finished matches as next week's positions.

    Tolerates fixtures_raw.csv being missing entirely (fetch_current_data's own fetch of it can
    fail - e.g. a transient 503 from football-data.co.uk - and a GitHub Actions runner starts from
    a fresh checkout each run, so there's no previous copy left on disk to fall back to the way a
    persistent machine would have). An empty fixture list for one run is a far smaller problem
    than the whole pipeline crashing here and never reaching match_archive further down, which is
    what tracked bets actually need graded."""
    try:
        df = pd.read_csv(os.path.join(WORKDIR, "fixtures_raw.csv"), encoding='utf-8-sig', on_bad_lines='skip')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Every column anything downstream selects from this DataFrame (see __main__'s
        # fd_fixtures[[...]] fallback path) - falls through the same processing below rather than
        # returning early, so Date ends up the same datetime64 dtype a real fetch would produce
        # (an early return with a raw object-dtype stub column would KeyError/AttributeError the
        # moment something downstream calls .dt.strftime() on it).
        df = pd.DataFrame(columns=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A'])
    e0 = df[df['Div'] == 'E0'].copy()
    e0['Date'] = pd.to_datetime(e0['Date'], format='mixed', dayfirst=True, errors='coerce')
    e0 = e0.dropna(subset=['Date', 'HomeTeam', 'AwayTeam']).sort_values('Date').reset_index(drop=True)

    if results is not None and current_season is not None:
        played = played_pairs_for_season(results, current_season)
        already_played = e0.apply(lambda r: (r['HomeTeam'], r['AwayTeam']) in played, axis=1)
        e0 = e0[~already_played].reset_index(drop=True)

    return e0


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
                'clean_sheet': (row['FTAG'] == 0) if is_home else (row['FTHG'] == 0),
            }
            team_games.setdefault(team, []).append(stats)
    return team_games


def build_team_history(results, window=ROLLING_WINDOW, use_last_n_seasons=2,
                        results_fallback=None, min_games=6, current_season=None, venue=None):
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
    caveat is visible downstream.

    `venue`: None (default) uses every game regardless of home/away, exactly as before. 'home' or
    'away' filters each team's own games down to that venue BEFORE the same bridging/windowing
    logic runs, so e.g. venue='home' builds "last `window` home games" using the identical
    warm-up/fallback rules as the blended version - just applied to a pre-filtered game list."""
    seasons_sorted = sorted(results['season'].unique())
    if current_season is None:
        current_season = seasons_sorted[-1]
    prior_seasons = [s for s in seasons_sorted[-use_last_n_seasons:] if s != current_season]

    cur_results = results[results['season'] == current_season].sort_values('Date')
    prior_results = results[results['season'].isin(prior_seasons)].sort_values('Date') if prior_seasons else results.iloc[0:0]

    def _venue_filter(games_by_team):
        if venue is None:
            return games_by_team
        want_home = (venue == 'home')
        return {t: [g for g in gs if g['is_home'] == want_home] for t, gs in games_by_team.items()}

    cur_games = _venue_filter(_extract_team_games(cur_results, 'Premier League'))
    prior_games = _venue_filter(_extract_team_games(prior_results, 'Premier League'))

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
        fb_team_games = _venue_filter(_extract_team_games(fb_recent, 'Championship'))

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


def team_trend(team_games, team, stat, window, league_avg):
    # No [-window:] slice here - build_team_history already decided the right sample (exactly
    # `window` during the early-season bridge, or all of the current season once that's >= window).
    # No fixed line either - `values` (the raw per-game numbers) is what lets matchup_engine.py
    # compute a hit rate at whatever line it ends up recommending for a specific matchup.
    games = team_games.get(team, [])
    vals = [g[stat] for g in games if g.get(stat) is not None and not (isinstance(g[stat], float) and np.isnan(g[stat]))]
    if len(vals) < 4:
        return None
    avg = np.mean(vals)
    plus_minus = avg - league_avg
    return {'n': len(vals), 'avg': round(float(avg), 2), 'plus_minus': round(float(plus_minus), 2),
            'league_avg': round(float(league_avg), 2), 'values': [round(float(v), 2) for v in vals]}


def build_all_trends(team_games, league_avgs, window=ROLLING_WINDOW):
    metrics = ['goals_for', 'goals_against', 'corners_for', 'corners_against',
               'cards_for', 'match_total_cards', 'shots_on_target_for', 'shots_on_target_against']

    trends = {}
    for team in team_games:
        trends[team] = {}
        for m in metrics:
            t = team_trend(team_games, team, m, window, league_avgs[m])
            if t:
                trends[team][m] = t

        games = team_games[team]
        if len(games) >= 4:
            cs_vals = [g['clean_sheet'] for g in games]
            trends[team]['clean_sheet'] = {'n': len(cs_vals), 'hit_rate': round(float(np.mean(cs_vals)) * 100, 1)}

        n_champ = sum(1 for g in games if g.get('division') == 'Championship')
        if n_champ > 0:
            trends[team]['_data_note'] = f"Includes {n_champ} Championship game(s) - newly promoted team, limited top-flight sample"

        # raw per-game log for the dashboard's per-team detail view - the exact games the trend
        # above is built from, newest first, so clicking into a fixture can show real match-by-
        # match history rather than just the aggregate numbers.
        trends[team]['_games'] = [{
            'date': g['date'].strftime('%Y-%m-%d'), 'opponent': g['opponent'],
            'is_home': bool(g['is_home']), 'division': g['division'],
            'goals_for': g['goals_for'], 'goals_against': g['goals_against'],
            'corners_for': g['corners_for'], 'corners_against': g['corners_against'],
            'cards_for': g['cards_for'], 'shots_on_target_for': g['shots_on_target_for'],
            'shots_on_target_against': g['shots_on_target_against'],
        } for g in sorted(games, key=lambda g: g['date'], reverse=True)]
    return trends


if __name__ == '__main__':
    print("Fetching latest data...")
    fetch_current_data()

    results = load_results('E0')
    results_championship = load_results('E1')
    print(f"Loaded {len(results)} PL results (latest: {results['Date'].max()}), "
          f"{len(results_championship)} Championship results (fallback for promoted teams)")
    current_season = current_season_code()
    fd_fixtures = load_fixtures(results, current_season)
    season_start_year = int(f"20{current_season[:2]}")
    played_pairs = played_pairs_for_season(results, current_season)
    # How far behind "now" football-data.co.uk's own results are - normally 0-1 days, but can be a
    # week or more if fetch_current_data() just fell back to a cached season file during an outage.
    # Widening ESPN's window to match means "played" stays accurate (round numbers, "what's next")
    # regardless of how long football-data.co.uk itself has been stale, rather than just covering
    # the handful of days a healthy week assumes.
    days_stale = max(6, (datetime.date.today() - results['Date'].max().date()).days + 2)
    espn_played = fetch_espn_recent_played_pairs(days_back=days_stale)
    played_pairs = played_pairs | espn_played
    print(f"Played-pairs freshness: {len(espn_played)} completed match(es) from ESPN "
          f"(last {days_stale}d) merged in on top of football-data.co.uk's own results")
    of_fixtures = fetch_openfootball_next_round(season_start_year, played_pairs)
    match_rounds, current_round = fetch_openfootball_rounds(season_start_year, played_pairs)
    print(f"Round mapping: {len(match_rounds)} fixtures across the season"
          + (f", current round {current_round}" if current_round else " (openfootball unreachable - tracker round grouping will be unavailable)"))

    if of_fixtures:
        # enrich with odds from football-data.co.uk's feed when that same fixture is in it
        odds_lookup = {(r['HomeTeam'], r['AwayTeam']): r for _, r in fd_fixtures.iterrows()}
        for fx in of_fixtures:
            odds_row = odds_lookup.get((fx['HomeTeam'], fx['AwayTeam']))
            for col in ('B365H', 'B365D', 'B365A'):
                fx[col] = odds_row.get(col) if odds_row is not None else None
        fixtures_out = of_fixtures
        print(f"Loaded {len(fixtures_out)} upcoming fixtures (openfootball, next round)")
    else:
        fixtures_out = fd_fixtures[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']].assign(
            Date=lambda d: d['Date'].dt.strftime('%Y-%m-%d')).to_dict('records')
        print(f"Loaded {len(fixtures_out)} upcoming fixtures "
              f"(football-data.co.uk fallback - openfootball unreachable)")

    team_games = build_team_history(results, results_fallback=results_championship,
                                     current_season=current_season)
    team_games_home = build_team_history(results, results_fallback=results_championship,
                                          current_season=current_season, venue='home')
    team_games_away = build_team_history(results, results_fallback=results_championship,
                                          current_season=current_season, venue='away')
    league_avgs = league_averages_from_matches(results)
    trends = build_all_trends(team_games, league_avgs)
    trends_home = build_all_trends(team_games_home, league_avgs)
    trends_away = build_all_trends(team_games_away, league_avgs)
    print(f"\nComputed trends for {len(trends)} teams "
          f"(+ home/away-split variants for the fixture card toggle)")
    print(f"League averages: {league_avgs}")

    out = {'generated_at': datetime.datetime.now().isoformat(), 'league_avgs': league_avgs,
           'trends': trends, 'trends_home': trends_home, 'trends_away': trends_away,
           'fixtures': fixtures_out, 'match_rounds': match_rounds, 'current_round': current_round}
    with open(os.path.join(WORKDIR, 'trends_data.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print("Saved trends_data.json")
