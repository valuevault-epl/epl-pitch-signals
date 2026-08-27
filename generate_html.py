import json
import os
import datetime

WORKDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WORKDIR, 'trends_data.json')) as f:
    data = json.load(f)

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


# dashboard.html: the Claude Artifact target (gitignored - published directly, not served from
# git). webapp_template.html -> docs/index.html: the GitHub Pages target, with real Supabase
# auth/sync - has to be a real hosted page since an Artifact's CSP blocks the external calls
# Supabase needs, so this is a genuinely separate build, not just a copy.
render(os.path.join(WORKDIR, 'dashboard_template.html'), os.path.join(WORKDIR, 'dashboard.html'))
render(os.path.join(WORKDIR, 'webapp_template.html'), os.path.join(WORKDIR, 'docs', 'index.html'))
