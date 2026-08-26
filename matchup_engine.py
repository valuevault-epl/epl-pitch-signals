"""
Combine each fixture's two teams' individual trends into matchup-specific, strength-adjusted
projections. Two teams' raw averages aren't directly comparable on their own - a team's own
average can be inflated or deflated just by the strength of opponents it happened to face
recently, and mixing two raw averages (old approach) inherits that distortion. Instead: rate
each team's own recent rate AGAINST the league average for that market (a "strength" ratio -
1.0 = exactly average, 1.3 = 30% above average, 0.7 = 30% below), then combine attack strength
x defense strength x league average - the standard expected-goals-style approach used throughout
football analytics - to get a projection adjusted for how strong each side actually is, not just
raw recent counting stats.

Lines are not fixed constants. Bookmakers offer alt lines for these markets - pretty much any
reasonable line exists - so instead of testing one hardcoded line (5.5 corners, 1.5 goals, etc.)
against every fixture, each signal recommends its OWN line: the projection, pushed a bit further
away in the safe direction by LEEWAY, then rounded to a bookmaker-style X.5 value. The hit rate
shown is each team's own history AT that recommended line, computed straight from their raw
per-game values (not a fixed-line lookup).

"edge" (projection vs league average, always defined and always genuinely comparable across
fixtures) drives sorting/bar-width/sign - NOT vs_line - because a recommended line is
constructed to sit a roughly constant buffer away from the projection, so vs_line stays close to
±LEEWAY regardless of how strong the underlying signal actually is and would be useless for
ranking fixtures by "how strong is this edge".
"""
import json
import math
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))

# How far below (OVER) or above (UNDER) the raw projection to set the recommended line, in the
# market's own units - a small buffer so the line isn't sitting right on top of the point
# estimate. Tuned per market to its typical scale.
LEEWAY = {
    'goals_for': 0.5, 'goals_against': 0.5,
    'corners_for': 1.5, 'corners_against': 1.5,
    'shots_on_target_for': 1.0, 'shots_on_target_against': 1.0,
    'match_cards': 1.0,
}


def recommend_line(projection, baseline, leeway):
    """Direction comes from projection vs baseline (the market's league average - the same
    comparison a fixed line near the league average used to encode implicitly). The line itself
    is always an X.5 value (bookmakers use half-lines to avoid pushes) derived from the
    projection with `leeway` built in as a safety buffer."""
    direction = 'OVER' if projection >= baseline else 'UNDER'
    if direction == 'OVER':
        target = projection - leeway
        line = math.floor(target * 2) / 2
        if line == int(line):
            line -= 0.5
    else:
        target = projection + leeway
        line = math.ceil(target * 2) / 2
        if line == int(line):
            line += 0.5
    return max(line, 0.5), direction


def hit_rate_at(values, line, direction):
    """% of a team's own raw per-game values that support the given call at the given line -
    computed fresh per matchup since the line itself is now matchup-specific, not a fixed lookup."""
    if not values:
        return None
    hits = sum(1 for v in values if (v > line if direction == 'OVER' else v < line))
    return round(hits / len(values) * 100, 1)


def combine_matchup(team_trends, opponent_trends, team_market, opponent_market, leeway):
    """team_market is the market this signal is ABOUT (e.g. goals_for, for 'this team to
    score'); opponent_market is the opponent's corresponding market (e.g. goals_against).
    Keys are named team_/opponent_, never home_/away_, so the caller can't mislabel who's who
    regardless of which side is actually home or away in the fixture."""
    t = team_trends.get(team_market)
    o = opponent_trends.get(opponent_market)
    if not t or not o:
        return None
    attack_strength = t['avg'] / t['league_avg'] if t['league_avg'] else 1.0
    defense_strength = o['avg'] / o['league_avg'] if o['league_avg'] else 1.0
    projection = t['league_avg'] * attack_strength * defense_strength
    line, direction = recommend_line(projection, t['league_avg'], leeway)
    return {
        'projection': round(projection, 2),
        'line': line,
        'direction': direction,
        'edge': round(projection - t['league_avg'], 2),
        'attack_strength': round(attack_strength, 2),
        'defense_strength': round(defense_strength, 2),
        'team_avg': t['avg'], 'team_hit_rate': hit_rate_at(t['values'], line, direction), 'team_n': t['n'],
        'opponent_avg': o['avg'], 'opponent_hit_rate': hit_rate_at(o['values'], line, direction), 'opponent_n': o['n'],
    }


def build_fixture_signals(fixture, trends):
    home, away = fixture['HomeTeam'], fixture['AwayTeam']
    home_trends = trends.get(home, {})
    away_trends = trends.get(away, {})
    if not home_trends or not away_trends:
        return None

    signals = {}
    hg = combine_matchup(home_trends, away_trends, 'goals_for', 'goals_against', LEEWAY['goals_for'])
    if hg:
        signals['home_goals'] = {'label': f'{home} to score', **hg}
    ag = combine_matchup(away_trends, home_trends, 'goals_for', 'goals_against', LEEWAY['goals_for'])
    if ag:
        signals['away_goals'] = {'label': f'{away} to score', **ag}
    hc = combine_matchup(home_trends, away_trends, 'corners_for', 'corners_against', LEEWAY['corners_for'])
    if hc:
        signals['home_corners'] = {'label': f'{home} corners', **hc}
    ac = combine_matchup(away_trends, home_trends, 'corners_for', 'corners_against', LEEWAY['corners_for'])
    if ac:
        signals['away_corners'] = {'label': f'{away} corners', **ac}
    hs = combine_matchup(home_trends, away_trends, 'shots_on_target_for', 'shots_on_target_against',
                          LEEWAY['shots_on_target_for'])
    if hs:
        signals['home_sot'] = {'label': f'{home} shots on target', **hs}
    as_ = combine_matchup(away_trends, home_trends, 'shots_on_target_for', 'shots_on_target_against',
                           LEEWAY['shots_on_target_for'])
    if as_:
        signals['away_sot'] = {'label': f'{away} shots on target', **as_}

    # match cards: projection sums each team's own cards-per-game rate (a card-proneness proxy -
    # not "conceded" by the other side, so no opponent blend for the number). Direction/line come
    # from that summed projection vs the league's match_total_cards average; the displayed hit
    # rate uses each team's own match_total_cards history (both teams' cards combined, in THEIR
    # games) at that recommended line - the true frequency of the total-cards market, not a
    # proxy built from a one-sided card count.
    hc_cards = home_trends.get('cards_for')
    ac_cards = away_trends.get('cards_for')
    hc_total = home_trends.get('match_total_cards')
    ac_total = away_trends.get('match_total_cards')
    if hc_cards and ac_cards and hc_total and ac_total:
        home_card_strength = hc_cards['avg'] / hc_cards['league_avg'] if hc_cards['league_avg'] else 1.0
        away_card_strength = ac_cards['avg'] / ac_cards['league_avg'] if ac_cards['league_avg'] else 1.0
        projection = hc_cards['avg'] + ac_cards['avg']
        league_total_cards = hc_total['league_avg']
        line, direction = recommend_line(projection, league_total_cards, LEEWAY['match_cards'])
        signals['match_cards'] = {
            'label': 'Total match cards',
            'projection': round(projection, 2), 'line': line, 'direction': direction,
            'edge': round(projection - league_total_cards, 2),
            'home_card_strength': round(home_card_strength, 2), 'away_card_strength': round(away_card_strength, 2),
            'home_team_avg': hc_cards['avg'],
            'home_team_hit_rate': hit_rate_at(hc_total['values'], line, direction), 'home_team_n': hc_total['n'],
            'away_team_avg': ac_cards['avg'],
            'away_team_hit_rate': hit_rate_at(ac_total['values'], line, direction), 'away_team_n': ac_total['n'],
        }

    return signals


if __name__ == '__main__':
    with open(os.path.join(WORKDIR, 'trends_data.json')) as f:
        data = json.load(f)

    fixtures_with_signals = []
    for fx in data['fixtures']:
        signals = build_fixture_signals(fx, data['trends'])
        fixtures_with_signals.append({**fx, 'signals': signals})

    data['fixtures'] = fixtures_with_signals
    with open(os.path.join(WORKDIR, 'trends_data.json'), 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Built matchup signals for {len(fixtures_with_signals)} fixtures")
    if fixtures_with_signals:
        fx0 = fixtures_with_signals[0]
        print(f"\nExample - {fx0['HomeTeam']} vs {fx0['AwayTeam']}:")
        for k, v in (fx0['signals'] or {}).items():
            print(f"  {k}: {v}")
    else:
        print("(no upcoming fixtures currently published by the data source)")
