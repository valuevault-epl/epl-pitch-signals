import json
import os
import datetime

WORKDIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WORKDIR, 'trends_data.json')) as f:
    data = json.load(f)

TEMPLATE_PATH = os.path.join(WORKDIR, 'dashboard_template.html')
OUT_PATH = os.path.join(WORKDIR, 'dashboard.html')

with open(TEMPLATE_PATH, encoding='utf-8') as f:
    template = f.read()

data_json = json.dumps(data, default=str)
generated_str = datetime.datetime.now().strftime('%A %d %B %Y, %H:%M')
html = template.replace('__DATA_JSON__', data_json).replace('__GENERATED_AT__', generated_str)

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Generated {OUT_PATH} ({len(html)} bytes)")
