import urllib.request
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# football-data.co.uk season codes: "0001" = 2000/01 ... "2526" = 2025/26
seasons = [f"{y%100:02d}{(y+1)%100:02d}" for y in range(2000, 2026)]

os.makedirs(os.path.join(WORKDIR, "seasons"), exist_ok=True)
for season in seasons:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    out_path = os.path.join(WORKDIR, "seasons", f"E0_{season}.csv")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        if len(data) < 500:
            print(f"{season}: too small, skipping ({len(data)} bytes)")
            continue
        with open(out_path, 'wb') as f:
            f.write(data)
        print(f"{season}: {len(data)} bytes")
    except Exception as e:
        print(f"{season}: FAILED - {e}")
