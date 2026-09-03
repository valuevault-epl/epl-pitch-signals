"""
Player-level trends and matchup signals - goals, shots, shots on target, assists (no cards:
Understat only has season-total cards, not per-match, so a real rolling-window cards market isn't
buildable from this source - see player_data.py).

Window is 30 games, not the 10 team trends use. A team's recent form window has to stay short
because the TEAM changes - transfers, tactical shifts, a new manager. A player's own skill is
comparatively stable game to game even as their team's makeup changes around them, so a longer
window trades a little recency for a lot more sample - the whole reason for a bigger window here.
Unlike team trends, this doesn't bridge across a season boundary or reset each season: it's a
simple trailing window over the player's most recent appearances (with actual minutes played),
regardless of which season or which club they were at - the same justification (individual skill
persists) applies there too.

No natural "league average player" baseline exists the way team markets have one (a striker's
shot rate and a defensive midfielder's aren't comparable) - so there's no attack/defense-strength
framing here. Instead: the player's own rolling average, adjusted by the upcoming opponent's team-
level defensive strength (reusing the SAME numbers trend_engine.py already computes for team
markets - shots/assists don't have an exact team-level analog, so shots_on_target_against and
goals_against are used as the closest available proxies, noted inline). Anchor lines and the
Poisson fair-odds model are the same approach as team markets (see matchup_engine.py, imported
directly rather than reimplemented).
"""
import json
import os

from matchup_engine import hit_rate_at, modeled_fair_odds, LINE_STEP

WORKDIR = os.path.dirname(os.path.abspath(__file__))

PLAYER_WINDOW = 30      # trailing games with actual minutes played, not adaptive/season-bound
MIN_PLAYER_GAMES = 8    # below this, too little data to trust - skip the player for that market

# Anchor line and how far the alt-line ladder is allowed to stretch, per player market.
PLAYER_ANCHOR = {'goals': 0.5, 'shots': 1.5, 'shots_on_target': 0.5, 'assists': 0.5}
PLAYER_ALT_RANGE = {'goals': 1.5, 'shots': 2.0, 'shots_on_target': 1.5, 'assists': 1.0}
PLAYER_MARKET_LABEL = {'goals': 'to score', 'shots': 'shots', 'shots_on_target': 'shots on target',
                        'assists': 'to assist'}

# Which of the OPPONENT's existing team-level defensive numbers to use as this player market's
# matchup adjustment. shots_on_target is an exact match to the team-level stat (both count the
# same thing); shots (total attempts) and assists don't have an exact team-level equivalent, so
# those two lean on the closest real proxies (shots_on_target_against correlates with shot volume
# conceded generally; goals_against correlates with the kind of chance quality that produces
# assists) rather than a precise match.
OPPONENT_DEFENSE_MARKET = {
    'goals': 'goals_against',
    'shots': 'shots_on_target_against',
    'shots_on_target': 'shots_on_target_against',
    'assists': 'goals_against',
}


def player_trend(matches, window=PLAYER_WINDOW):
    """matches: Understat's per-match log for one player, most recent first (their own API
    already orders it that way). Only counts appearances with actual minutes - an unused sub
    shouldn't drag a rate down to 0 for a game they didn't really play."""
    played = [m for m in matches if float(m.get('time') or 0) > 0]
    recent = played[:window]
    if len(recent) < MIN_PLAYER_GAMES:
        return None
    stats = {}
    for market, field in (('goals', 'goals'), ('shots', 'shots'), ('shots_on_target', 'shots_on_target'),
                           ('assists', 'assists')):
        values = [float(m[field]) for m in recent]
        stats[market] = {'avg': round(sum(values) / len(values), 2), 'values': values, 'n': len(values)}
    return stats


# UNDER calls on every player market are almost always the trivially-true, low-value kind (a
# defender or keeper "under 0.5 goals/shots/assists" hits ~100% but prices near 1.01 - no real
# signal, just noise crowding out the Strong list). Every current player market is a low counting
# stat with this same lopsided shape, so all of them are OVER-only.
OVER_ONLY_MARKETS = {'goals', 'shots', 'shots_on_target', 'assists'}


def build_player_signal(player_name, team_title, trend, market, opponent_team_trends):
    """One player, one market (goals/shots/assists), against a specific opponent. Mirrors
    combine_matchup's shape (line/direction/edge/modeled_odds/hit_rate/alt_lines) so the dashboard
    can reuse the same rendering code for player signals as team ones."""
    stat = trend.get(market) if trend else None
    if not stat:
        return None
    defense_key = OPPONENT_DEFENSE_MARKET[market]
    opp_stat = (opponent_team_trends or {}).get(defense_key)
    defense_strength = (opp_stat['avg'] / opp_stat['league_avg']) if opp_stat and opp_stat.get('league_avg') else 1.0
    projection = stat['avg'] * defense_strength

    anchor = PLAYER_ANCHOR[market]
    direction = 'OVER' if projection >= anchor else 'UNDER'
    if market in OVER_ONLY_MARKETS and direction == 'UNDER':
        return None
    line = anchor

    alt_range = PLAYER_ALT_RANGE[market]
    steps = int(round(alt_range / LINE_STEP))
    alt_lines = []
    seen = set()
    for i in range(-steps, steps + 1):
        al = round(anchor + i * LINE_STEP, 1)
        if al == int(al) or al <= 0 or al in seen:
            continue
        seen.add(al)
        alt_lines.append({
            'line': al,
            'direction': direction,  # always OVER (see OVER_ONLY_MARKETS) - kept for shape
                                      # parity with team alt_lines, which now carry both directions
            'hit_rate': hit_rate_at(stat['values'], al, direction),
            'modeled_odds': modeled_fair_odds(al, direction, projection),
        })

    return {
        'player': player_name, 'team': team_title,
        'label': f'{player_name} {PLAYER_MARKET_LABEL[market]}',
        'market': market,
        'projection': round(projection, 2), 'line': line, 'direction': direction,
        'anchor': anchor, 'alt_lines': alt_lines,
        'defense_strength': round(defense_strength, 2),
        'own_avg': stat['avg'], 'hit_rate': hit_rate_at(stat['values'], line, direction), 'n': stat['n'],
        'modeled_odds': modeled_fair_odds(line, direction, projection),
    }
