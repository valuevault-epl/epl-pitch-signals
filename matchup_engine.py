"""
Combine each fixture's two teams' individual trends into matchup-specific projections: e.g.
Man City's own average corners blended with Coventry's own average corners-conceded, projecting
what THIS specific matchup should produce - not a comparison against an abstract league average.
The projection is then measured against the actual betting line for that market (e.g. 5.5
corners), so the +/- number directly means "how far above/below the typical line we project."
"""
import json
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def combine_matchup(team_trends, opponent_trends, team_market, opponent_market):
    """team_market is the market this signal is ABOUT (e.g. goals_for, for 'this team to
    score'); opponent_market is the opponent's corresponding market (e.g. goals_against).
    Projection = average of the team's own recent rate and this specific opponent's own recent
    rate at allowing/conceding it - a matchup-specific blend, not a league-average comparison.
    Keys are named team_/opponent_, never home_/away_, so the caller can't mislabel who's who
    regardless of which side is actually home or away in the fixture."""
    t = team_trends.get(team_market)
    o = opponent_trends.get(opponent_market)
    if not t or not o:
        return None
    projection = (t['avg'] + o['avg']) / 2
    return {
        'projection': round(projection, 2),
        'line': t['line'],
        'vs_line': round(projection - t['line'], 2),
        'team_avg': t['avg'], 'team_hit_rate': t['hit_rate'], 'team_n': t['n'],
        'opponent_avg': o['avg'], 'opponent_hit_rate': o['hit_rate'], 'opponent_n': o['n'],
        'avg_hit_rate': round((t['hit_rate'] + o['hit_rate']) / 2, 1),
    }


def build_fixture_signals(fixture, trends):
    home, away = fixture['HomeTeam'], fixture['AwayTeam']
    home_trends = trends.get(home, {})
    away_trends = trends.get(away, {})
    if not home_trends or not away_trends:
        return None

    signals = {}
    hg = combine_matchup(home_trends, away_trends, 'goals_for', 'goals_against')
    if hg:
        signals['home_goals'] = {'label': f'{home} to score', **hg}
    ag = combine_matchup(away_trends, home_trends, 'goals_for', 'goals_against')
    if ag:
        signals['away_goals'] = {'label': f'{away} to score', **ag}
    hc = combine_matchup(home_trends, away_trends, 'corners_for', 'corners_against')
    if hc:
        signals['home_corners'] = {'label': f'{home} corners', **hc}
    ac = combine_matchup(away_trends, home_trends, 'corners_for', 'corners_against')
    if ac:
        signals['away_corners'] = {'label': f'{away} corners', **ac}
    hs = combine_matchup(home_trends, away_trends, 'shots_on_target_for', 'shots_on_target_against')
    if hs:
        signals['home_sot'] = {'label': f'{home} shots on target', **hs}
    as_ = combine_matchup(away_trends, home_trends, 'shots_on_target_for', 'shots_on_target_against')
    if as_:
        signals['away_sot'] = {'label': f'{away} shots on target', **as_}

    # match cards: both teams contribute their own average directly to the match total, so the
    # projection is simply the sum of each team's own recent cards-per-game rate (no opponent
    # blend needed here - a team's card tendency isn't really "conceded" by the other side).
    hc_cards = home_trends.get('cards_for')
    ac_cards = away_trends.get('cards_for')
    if hc_cards and ac_cards:
        match_cards_line = 3.5  # standard "total match cards" betting line
        projection = hc_cards['avg'] + ac_cards['avg']
        signals['match_cards'] = {
            'label': 'Total match cards',
            'projection': round(projection, 2), 'line': match_cards_line,
            'vs_line': round(projection - match_cards_line, 2),
            'home_team_avg': hc_cards['avg'], 'home_team_hit_rate': hc_cards['hit_rate'], 'home_team_n': hc_cards['n'],
            'away_team_avg': ac_cards['avg'], 'away_team_hit_rate': ac_cards['hit_rate'], 'away_team_n': ac_cards['n'],
            'avg_hit_rate': round((hc_cards['hit_rate'] + ac_cards['hit_rate']) / 2, 1),
        }

    # btts: both teams' individual btts hit rates (already a probability-style read, no line)
    if 'btts' in home_trends and 'btts' in away_trends:
        signals['btts'] = {
            'label': 'Both teams to score',
            'home_hit_rate': home_trends['btts']['hit_rate'], 'home_n': home_trends['btts']['n'],
            'away_hit_rate': away_trends['btts']['hit_rate'], 'away_n': away_trends['btts']['n'],
            'avg_hit_rate': round((home_trends['btts']['hit_rate'] + away_trends['btts']['hit_rate']) / 2, 1),
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
    fx0 = fixtures_with_signals[0]
    print(f"\nExample - {fx0['HomeTeam']} vs {fx0['AwayTeam']}:")
    for k, v in (fx0['signals'] or {}).items():
        print(f"  {k}: {v}")
