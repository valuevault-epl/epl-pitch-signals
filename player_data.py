"""
Player-level match data from Understat - the only free source found with real per-match player
stats (goals, shots, assists, xG, minutes). football-data.co.uk (this dashboard's main source)
is team-only box scores; football-data.org's free tier explicitly excludes player data; StatsBomb
Open Data's Premier League coverage is two isolated old seasons (2003/04, 2015/16), not ongoing.

IMPORTANT CAVEAT: these are Understat's own internal AJAX endpoints (getLeagueData/,
getPlayerData/), reverse-engineered from their site's JS bundle - not a published, documented
API. They could change or start blocking automated access at any time without notice. Treated
accordingly: every request is wrapped so a failure degrades gracefully (skip that player/data,
don't crash the pipeline), responses are cached locally so a bad day doesn't lose everything
already fetched, and requests are politely rate-limited (SLEEP_BETWEEN_REQUESTS).

No per-match cards data exists in this source (only season totals), so player cards markets are
not built - goals, shots, shots on target, and assists are.

Which players to track for an upcoming fixture is itself an approximation: Understat has no
"predicted lineup" data, so this uses whichever players have the most recent appearances for that
team as a proxy for who's likely to play - doesn't know about injuries, suspensions, or last-
minute rotation.
"""
import gzip
import json
import os
import time
import urllib.request

WORKDIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(WORKDIR, "player_cache")
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://understat.com/',
}
SLEEP_BETWEEN_REQUESTS = 0.4  # politeness delay - this is an unofficial endpoint, not a public API
KEY_PLAYERS_PER_TEAM = 6      # most-recently-featured players per team to track, as a lineup proxy


def _fetch_json(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            # understat always gzips this endpoint regardless of Accept-Encoding - urllib doesn't
            # auto-decompress the way curl --compressed or a browser would, so this is explicit.
            if raw[:2] == b'\x1f\x8b':
                raw = gzip.decompress(raw)
            return json.loads(raw.decode('utf-8'))
    except Exception:
        return None


def fetch_roster(season):
    """season like '2025' for the 2025-26 season (Understat's season param is the start year).
    Returns {player_name: {'id', 'team_title', 'games'}} for everyone who played that season."""
    data = _fetch_json(f"https://understat.com/getLeagueData/EPL/{season}")
    if not data:
        return {}
    return {p['player_name']: {'id': p['id'], 'team_title': p['team_title'], 'games': int(p['games'])}
            for p in data.get('players', [])}


# On-target = would have gone in without a defensive save/deflection stopping it - Goal or
# SavedShot. ShotOnPost is deliberately excluded (it missed the frame, same convention
# football-data.co.uk's own HST/AST already use for team shots on target, so this stays
# consistent with the rest of the dashboard rather than introducing a different definition).
ON_TARGET_RESULTS = {'Goal', 'SavedShot'}


def fetch_player_data(player_id):
    """Understat's getPlayerData response in one request has both `matches` (per-match goals/
    shots/assists/minutes, most recent first - already how their site orders it) and `shots`
    (every individual shot, with an outcome per shot - Goal/SavedShot/ShotOnPost/MissedShots/
    BlockedShot - which `matches` doesn't break down). Grouping `shots` by match_id and counting
    ON_TARGET_RESULTS gives a genuine per-match shots-on-target count, the one stat `matches`
    doesn't already provide directly."""
    data = _fetch_json(f"https://understat.com/getPlayerData/{player_id}")
    if not data:
        return None
    matches = data.get('matches', [])
    sot_by_match = {}
    for s in data.get('shots', []):
        if s.get('result') in ON_TARGET_RESULTS:
            sot_by_match[s['match_id']] = sot_by_match.get(s['match_id'], 0) + 1
    for m in matches:
        m['shots_on_target'] = sot_by_match.get(m['id'], 0)
    return matches


def cached_player_matches(player_id, player_name):
    """Local cache keyed by player id - avoids re-fetching a player's whole match history (which
    can be hundreds of matches back to 2014+ for long careers) every single pipeline run. Re-fetch
    happens whenever this is called; caching here is just about not re-fetching the SAME player
    twice within one run (build_key_players can reference the same player across markets)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{player_id}.json")
    matches = fetch_player_data(player_id)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    if matches is None:
        # fetch failed - fall back to whatever was cached from a previous successful run, if any,
        # rather than losing this player entirely because of one bad request.
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    with open(path, 'w') as f:
        json.dump(matches, f)
    return matches


def key_players_for_team(current_roster, prior_roster, team_title, n=KEY_PLAYERS_PER_TEAM):
    """Team assignment comes from the CURRENT season's roster (accurate now, including summer
    transfers) - but early in a new season everyone's current-season appearance count is 0 or 1,
    which can't differentiate a regular starter from a fringe squad player. Ranked by PRIOR
    season's games played instead (falls back to 0 for a new signing/promotion, who'd then just
    rank low rather than being excluded outright) - a proxy for who's likely to feature, same
    spirit as the team-trend bridge from last season early in a new one, not a guarantee
    (injuries/suspensions/rotation aren't accounted for, and there's no predicted-lineup data).

    Understat's own team_title is comma-joined for a player who transferred mid-season (e.g.
    "Arsenal,Crystal Palace") - split on comma rather than exact-matching, so a mid-season
    transfer still counts toward both clubs instead of neither."""
    team_players = [(name, info) for name, info in current_roster.items()
                     if team_title in info['team_title'].split(',')]
    team_players.sort(key=lambda kv: prior_roster.get(kv[0], {}).get('games', 0), reverse=True)
    return team_players[:n]
