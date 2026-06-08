import json
from pathlib import Path

d = json.loads(Path("data/2788855626/analysis.json").read_text())
peaks = [p for p in d["loud_peaks"] if 1100 <= p["t"] <= 1240]
print(f"peaks {len(peaks)} between 1100-1240:")
for p in peaks:
    print(f"  {p['t']:7.1f}  score={p['score']:.2f}")
