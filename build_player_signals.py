"""
Builds player-market signals (goals/shots/assists) for every team in this week's fixtures and
merges them into trends_data.json as `player_signals` on each fixture. Run after trend_engine.py
(needs the team trends it writes, for the opponent-defense adjustment) and independently of
matchup_engine.py/market_floor.py - order between those doesn't matter.

Understat's own site labels a season by its start year (e.g. "2025" = 2025-26) - same convention
trend_engine.py already uses for football-data.co.uk season codes, kept consistent here.
"""
import datetime
import json
import os

from trend_engine import WORKDIR, current_season_code
from player_data import fetch_roster, cached_player_matches, key_players_for_team
from player_engine import player_trend, build_player_signal

PLAYER_MARKETS = ['goals', 'shots', 'shots_on_target', 'assists']


def build_all_player_signals(fixtures, team_trends):
    current_season = current_season_code()  # e.g. '2627' for 2026-27
    current_start_year = str(2000 + int(current_season[:2]))
    prior_start_year = str(int(current_start_year) - 1)

    print(f"Fetching Understat rosters: current={current_start_year}, prior={prior_start_year}...")
    current_roster = fetch_roster(current_start_year)
    prior_roster = fetch_roster(prior_start_year)
    if not current_roster:
        print("  Understat roster fetch failed or returned nothing - skipping player signals "
              "entirely this run (team/match signals are unaffected).")
        return fixtures, {}

    teams_needed = {fx['HomeTeam'] for fx in fixtures} | {fx['AwayTeam'] for fx in fixtures}
    # football-data.co.uk and Understat don't always spell team names the same way - map by
    # matching whichever Understat team_title starts with or is contained in our team name, since
    # exact-string mismatches (e.g. "Man United" vs "Manchester United") would otherwise silently
    # drop that team's players entirely.
    understat_teams = {info['team_title'] for info in current_roster.values()}
    # Understat's team_title can be a comma-joined list for a player who transferred clubs
    # mid-season (e.g. "Arsenal,Crystal Palace") - splitting on comma so those still match either
    # club correctly instead of only matching whichever one happens to substring-match first.
    understat_teams_split = set()
    for ut in understat_teams:
        understat_teams_split.update(ut.split(','))
    # Explicit overrides for names substring-matching can't bridge (abbreviations, no shared
    # substring at all) - "Nott'm Forest" vs "Nottingham Forest" confirmed as exactly this case.
    EXPLICIT_MAP = {"Nott'm Forest": "Nottingham Forest"}

    def match_understat_name(team):
        if team in EXPLICIT_MAP:
            return EXPLICIT_MAP[team]
        if team in understat_teams_split:
            return team
        for ut in understat_teams_split:
            if team.replace("Man ", "Manchester ") == ut or team in ut or ut in team:
                return ut
        return None

    team_name_map = {}
    for t in teams_needed:
        u = match_understat_name(t)
        if u:
            team_name_map[t] = u
        else:
            print(f"  Could not match '{t}' to an Understat team name - no player signals for them this week.")

    print(f"Fetching player match logs for {len(team_name_map)} teams "
          f"(~{len(team_name_map) * 6} players, rate-limited)...")
    trend_cache = {}  # player_name -> player_trend() result
    # Raw per-match records (goals/shots/shots_on_target/assists + date), kept separately from the
    # windowed trend above - this is what lets a tracked bet on a player leg still be gradeable
    # once its match has been played, the same reasoning as market_floor.py's team match archive.
    # Understat's own `date`/`h_team`/`a_team` fields on each match are enough to build the key.
    player_match_archive = {}
    for team, u_team in team_name_map.items():
        for name, info in key_players_for_team(current_roster, prior_roster, u_team):
            if name in trend_cache:
                continue
            matches = cached_player_matches(info['id'], name)
            trend_cache[name] = player_trend(matches) if matches else None
            # Capped to the current + prior season, same span the live trend window mostly draws
            # from - a tracked bet is realistically graded within days of its match being played,
            # so a player's ENTIRE career history (Understat goes back to 2014) is far more than
            # this needs and was bloating the embedded JSON by well over a megabyte for no benefit.
            for m in (matches or []):
                if float(m.get('time') or 0) <= 0 or m.get('season') not in (current_start_year, prior_start_year):
                    continue
                key = f"{name}|{m['date']}"
                player_match_archive[key] = {
                    'goals': float(m['goals']), 'shots': float(m['shots']),
                    'shots_on_target': float(m.get('shots_on_target', 0)), 'assists': float(m['assists']),
                }

    for fx in fixtures:
        signals = []
        for side, team, opponent in (('H', fx['HomeTeam'], fx['AwayTeam']), ('A', fx['AwayTeam'], fx['HomeTeam'])):
            u_team = team_name_map.get(team)
            if not u_team:
                continue
            opponent_trends = team_trends.get(opponent, {})
            for name, info in key_players_for_team(current_roster, prior_roster, u_team):
                trend = trend_cache.get(name)
                if not trend:
                    continue
                for market in PLAYER_MARKETS:
                    sig = build_player_signal(name, team, trend, market, opponent_trends)
                    if sig:
                        sig['side'] = side
                        signals.append(sig)
        fx['player_signals'] = signals

    total = sum(len(fx['player_signals']) for fx in fixtures)
    print(f"Built {total} player signals across {len(fixtures)} fixtures.")
    print(f"Player match archive: {len(player_match_archive)} match records.")
    return fixtures, player_match_archive


if __name__ == '__main__':
    trends_path = os.path.join(WORKDIR, 'trends_data.json')
    with open(trends_path) as f:
        data = json.load(f)

    data['fixtures'], data['player_match_archive'] = build_all_player_signals(data['fixtures'], data['trends'])

    with open(trends_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Merged player_signals, player_match_archive into {trends_path}")
