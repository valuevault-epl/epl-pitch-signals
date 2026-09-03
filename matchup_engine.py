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

Lines are anchored at the STANDARD main line real bookmakers actually use for each market
(ANCHOR_LINE below), not derived from the projection. Real books mostly keep these standardized
across teams rather than moving the line per team - confirmed directly for corners: a heavy-
corner side like Man City still gets offered 5.5, not a team-adjusted line, because price does
the work of reflecting team strength instead of the line moving. Earlier versions of this file
searched for a line that reached a target hit rate, pushing it further from the projection the
weaker a signal was - which drifted away from how real alt-line menus are actually built, and
could land on lines a bookmaker would price nowhere near what the search's formula assumed.

Since there's no live odds feed for these specific team-level alt-line markets (checked - nothing
free covers them), `modeled_odds` estimates an approximate FAIR price (no bookmaker margin) at
whatever line is showing, using a Poisson model of the matchup projection - not a replacement for
real prices, just a sanity-check estimate to compare against what's actually quoted. The dropdown
still lets a viewer explore alt lines around the anchor and see hit rate / modeled odds update for
each.

Cards specifically: real books usually price "booking points" (yellow=10, red=25), not a raw card
count the way this dashboard tracks cards - 3.5 is the conventional line for a raw count, but it
isn't a true match for how a bookmaker's actual cards market is structured. Shots-on-target
evidence on whether books keep one standard line per team was mixed, so treat 4.5 there as a
rougher estimate than the corners/goals/cards anchors.
"""
import json
import math
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))

# Standard main lines real bookmakers use for these team-level markets - see module docstring.
ANCHOR_LINE = {
    'goals_for': 1.5, 'goals_against': 1.5,
    'corners_for': 5.5, 'corners_against': 5.5,
    'shots_on_target_for': 4.5, 'shots_on_target_against': 4.5,
    'match_cards': 3.5,
}
ALT_RANGE = 2.0    # dashboard dropdown offers anchor +/- this many units, in 0.5 steps
LINE_STEP = 0.5


def hit_rate_at(values, line, direction):
    """% of a team's own raw per-game values that support the given call at the given line."""
    if not values:
        return None
    hits = sum(1 for v in values if (v > line if direction == 'OVER' else v < line))
    return round(hits / len(values) * 100, 1)


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_cdf(k, lam):
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(int(k) + 1))


def modeled_fair_odds(line, direction, lam):
    """Approximate no-margin fair odds at `line`, modeling the stat as Poisson(lam) - lam is the
    matchup projection, which already accounts for attack/defense strength. A real bookmaker price
    would sit BELOW this (their margin shortens it further), so treat this as an optimistic
    ceiling, not a number you'd expect to actually see quoted."""
    p = 1 - poisson_cdf(math.floor(line), lam) if direction == 'OVER' else poisson_cdf(math.floor(line), lam)
    p = max(min(p, 0.995), 0.005)  # guard against a 0%/100% model estimate producing a nonsense price
    return round(1 / p, 2)


def alt_lines_for(anchor, projection, team_values, opp_values, alt_range=ALT_RANGE):
    """Ladder of lines around the anchor, in BOTH directions at every line - lets the dashboard's
    per-signal dropdown offer Over AND Under at any distance from the anchor, rather than locking
    a viewer into whichever direction the model's own projection happened to favor. (Team markets
    only - player markets stay Over-only, see player_engine.py's OVER_ONLY_MARKETS for why an
    Under is excluded there entirely rather than just de-emphasized.)"""
    steps = int(round(alt_range / LINE_STEP))
    lines = []
    for i in range(-steps, steps + 1):
        line = round(anchor + i * LINE_STEP, 1)
        if line == int(line):
            continue  # bookmakers use half-lines, not whole numbers, to avoid pushes
        if line <= 0:
            continue
        for direction in ('OVER', 'UNDER'):
            lines.append({
                'line': line,
                'direction': direction,
                'team_hit_rate': hit_rate_at(team_values, line, direction),
                'opponent_hit_rate': hit_rate_at(opp_values, line, direction),
                'modeled_odds': modeled_fair_odds(line, direction, projection),
            })
    return lines


def combine_matchup(team_trends, opponent_trends, team_market, opponent_market, anchor):
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
    direction = 'OVER' if projection >= t['league_avg'] else 'UNDER'
    line = anchor
    return {
        'projection': round(projection, 2),
        'line': line,
        'direction': direction,
        'anchor': anchor,
        'alt_lines': alt_lines_for(anchor, projection, t['values'], o['values']),
        'team_market': team_market, 'opponent_market': opponent_market,
        'edge': round(projection - t['league_avg'], 2),
        'attack_strength': round(attack_strength, 2),
        'defense_strength': round(defense_strength, 2),
        'modeled_odds': modeled_fair_odds(line, direction, projection),
        'team_avg': t['avg'], 'team_hit_rate': hit_rate_at(t['values'], line, direction), 'team_n': t['n'],
        'opponent_avg': o['avg'], 'opponent_hit_rate': hit_rate_at(o['values'], line, direction), 'opponent_n': o['n'],
    }


def _signals_from_trends(fixture, home_trends, away_trends):
    """Build every signal for a fixture from a given pair of team-trend dicts. Factored out so the
    same logic can run twice per fixture: once on the blended (venue-agnostic) trends, and once on
    a venue-merged pair (home team's home-only numbers, away team's away-only numbers) for the
    dashboard's Overall / Home-Away toggle - see build_fixture_signals below."""
    home, away = fixture['HomeTeam'], fixture['AwayTeam']
    signals = {}
    hg = combine_matchup(home_trends, away_trends, 'goals_for', 'goals_against', ANCHOR_LINE['goals_for'])
    if hg:
        signals['home_goals'] = {'label': f'{home} to score', **hg}
    ag = combine_matchup(away_trends, home_trends, 'goals_for', 'goals_against', ANCHOR_LINE['goals_for'])
    if ag:
        signals['away_goals'] = {'label': f'{away} to score', **ag}
    hc = combine_matchup(home_trends, away_trends, 'corners_for', 'corners_against', ANCHOR_LINE['corners_for'])
    if hc:
        signals['home_corners'] = {'label': f'{home} corners', **hc}
    ac = combine_matchup(away_trends, home_trends, 'corners_for', 'corners_against', ANCHOR_LINE['corners_for'])
    if ac:
        signals['away_corners'] = {'label': f'{away} corners', **ac}
    hs = combine_matchup(home_trends, away_trends, 'shots_on_target_for', 'shots_on_target_against',
                          ANCHOR_LINE['shots_on_target_for'])
    if hs:
        signals['home_sot'] = {'label': f'{home} shots on target', **hs}
    as_ = combine_matchup(away_trends, home_trends, 'shots_on_target_for', 'shots_on_target_against',
                           ANCHOR_LINE['shots_on_target_for'])
    if as_:
        signals['away_sot'] = {'label': f'{away} shots on target', **as_}

    # match cards: projection sums each team's own cards-per-game rate (a card-proneness proxy -
    # not "conceded" by the other side, so no opponent blend for the number). Direction comes from
    # that summed projection vs the league's match_total_cards average; the displayed hit rate
    # uses each team's own match_total_cards history (both teams' cards combined, in THEIR games)
    # at the anchor line - the true frequency of the total-cards market, not a proxy built from a
    # one-sided card count.
    hc_cards = home_trends.get('cards_for')
    ac_cards = away_trends.get('cards_for')
    hc_total = home_trends.get('match_total_cards')
    ac_total = away_trends.get('match_total_cards')
    if hc_cards and ac_cards and hc_total and ac_total:
        home_card_strength = hc_cards['avg'] / hc_cards['league_avg'] if hc_cards['league_avg'] else 1.0
        away_card_strength = ac_cards['avg'] / ac_cards['league_avg'] if ac_cards['league_avg'] else 1.0
        projection = hc_cards['avg'] + ac_cards['avg']
        league_total_cards = hc_total['league_avg']
        direction = 'OVER' if projection >= league_total_cards else 'UNDER'
        line = ANCHOR_LINE['match_cards']
        signals['match_cards'] = {
            'label': 'Total match cards',
            'projection': round(projection, 2), 'line': line, 'direction': direction,
            'anchor': line,
            'alt_lines': alt_lines_for(line, projection, hc_total['values'], ac_total['values']),
            'team_market': 'match_total_cards', 'opponent_market': 'match_total_cards',
            'edge': round(projection - league_total_cards, 2),
            'modeled_odds': modeled_fair_odds(line, direction, projection),
            'home_card_strength': round(home_card_strength, 2), 'away_card_strength': round(away_card_strength, 2),
            'home_team_avg': hc_cards['avg'],
            'home_team_hit_rate': hit_rate_at(hc_total['values'], line, direction), 'home_team_n': hc_total['n'],
            'away_team_avg': ac_cards['avg'],
            'away_team_hit_rate': hit_rate_at(ac_total['values'], line, direction), 'away_team_n': ac_total['n'],
        }

    return signals


def _venue_merged_trends(blended_team_trends, venue_team_trends):
    """A team's blended trend dict, with any market overridden by its venue-specific (home-only or
    away-only) equivalent wherever one exists. A market missing from `venue_team_trends` (fewer
    than 4 valid games at that venue - see trend_engine.team_trend) just keeps its blended value,
    so the toggle degrades gracefully instead of losing signals for teams with a thin home/away
    sample so far this season."""
    merged = dict(blended_team_trends)
    for market, v in (venue_team_trends or {}).items():
        if not market.startswith('_') and v:
            merged[market] = v
    return merged


def build_fixture_signals(fixture, trends, trends_home=None, trends_away=None):
    """Returns (signals, signals_venue). `signals` is the existing blended-sample view. When
    trends_home/trends_away are supplied and both teams have at least some venue-specific data,
    `signals_venue` is a second, parallel signal set built with the home team's home-only numbers
    and the away team's away-only numbers wherever each team's data is used - as the subject of
    its own market AND as the opponent/defense context feeding the other team's market - instead
    of the blended last-N-regardless-of-venue sample. Otherwise `signals_venue` is None and the
    dashboard's toggle has nothing to switch to for that fixture."""
    home, away = fixture['HomeTeam'], fixture['AwayTeam']
    home_trends = trends.get(home, {})
    away_trends = trends.get(away, {})
    if not home_trends or not away_trends:
        return None, None

    signals = _signals_from_trends(fixture, home_trends, away_trends)

    signals_venue = None
    if trends_home is not None and trends_away is not None:
        home_venue = trends_home.get(home)
        away_venue = trends_away.get(away)
        if home_venue and away_venue:
            merged_home = _venue_merged_trends(home_trends, home_venue)
            merged_away = _venue_merged_trends(away_trends, away_venue)
            signals_venue = _signals_from_trends(fixture, merged_home, merged_away)

    return signals, signals_venue


if __name__ == '__main__':
    with open(os.path.join(WORKDIR, 'trends_data.json')) as f:
        data = json.load(f)

    trends_home = data.get('trends_home')
    trends_away = data.get('trends_away')

    fixtures_with_signals = []
    n_venue = 0
    for fx in data['fixtures']:
        signals, signals_venue = build_fixture_signals(fx, data['trends'], trends_home, trends_away)
        if signals_venue:
            n_venue += 1
        fixtures_with_signals.append({**fx, 'signals': signals, 'signals_venue': signals_venue})

    data['fixtures'] = fixtures_with_signals
    with open(os.path.join(WORKDIR, 'trends_data.json'), 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Built matchup signals for {len(fixtures_with_signals)} fixtures "
          f"({n_venue} with a home/away-split alternative)")
    if fixtures_with_signals:
        fx0 = fixtures_with_signals[0]
        print(f"\nExample - {fx0['HomeTeam']} vs {fx0['AwayTeam']}:")
        for k, v in (fx0['signals'] or {}).items():
            print(f"  {k}: {v}")
    else:
        print("(no upcoming fixtures currently published by the data source)")
