import json
import os
import re
import datetime

WORKDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WORKDIR, 'trends_data.json')) as f:
    data = json.load(f)


def _published_current_round(path):
    """The current_round already baked into a previously-published page, or None if the file
    doesn't exist yet or has none. Used by the regression guard below."""
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        content = f.read()
    m = re.search(r'"current_round":\s*(\d+)', content)
    return int(m.group(1)) if m else None


new_round = data.get('current_round')
old_round = _published_current_round(os.path.join(WORKDIR, 'docs', 'index.html'))

if isinstance(new_round, int) and isinstance(old_round, int) and new_round < old_round:
    # A round number can only ever advance as matches get played - never regress. Seeing one
    # anyway means THIS run's data sources are stuck behind what a previous run already
    # established - e.g. football-data.co.uk is down for this run too but ESPN's played-pairs
    # supplement (trend_engine.fetch_espn_recent_played_pairs) isn't reachable from here either
    # (blocked entirely on GitHub Actions - see ESPN_HEADERS' comment in trend_engine.py), while
    # an earlier run - a manual one from a residential connection, or a scheduled one that
    # happened to catch football-data.co.uk while it was briefly up - had enough to see those
    # matches as played and moved on. Publishing this would revert the live site back to older
    # fixtures/results a previous, better-informed run had already moved past - confirmed exactly
    # this happening repeatedly (a manual round-4 refresh getting silently reverted to round 3 by
    # the next few scheduled runs). Skipping leaves the already-published pages untouched, which
    # is strictly better than regressing; the next run just tries again from scratch.
    print(f"Skipping publish: newly computed current_round ({new_round}) is behind what's "
          f"already live ({old_round}) - this run's data sources are stuck behind a previous, "
          f"better-informed run. Leaving the published pages exactly as they are.")
else:
    data_json = json.dumps(data, default=str)
    generated_str = datetime.datetime.now().strftime('%A %d %B %Y, %H:%M')

    def render(template_path, out_path):
        with open(template_path, encoding='utf-8') as f:
            template = f.read()
        html = template.replace('__DATA_JSON__', data_json).replace('__GENERATED_AT__', generated_str)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Generated {out_path} ({len(html)} bytes)")

    # dashboard.html: the Claude Artifact target (published directly, but also committed here as
    # a mirror). webapp_template.html -> docs/index.html: the GitHub Pages target, with real
    # Supabase auth/sync - has to be a real hosted page since an Artifact's CSP blocks the
    # external calls Supabase needs, so this is a genuinely separate build, not just a copy.
    render(os.path.join(WORKDIR, 'dashboard_template.html'), os.path.join(WORKDIR, 'dashboard.html'))
    render(os.path.join(WORKDIR, 'webapp_template.html'), os.path.join(WORKDIR, 'docs', 'index.html'))
