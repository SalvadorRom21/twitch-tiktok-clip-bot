"""Summarize dense_hunt by high name / terminal scores and list saved JPGs."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/reference/clip_reviews/1fbc9fdef6a0_deep")
data = json.loads((OUT / "dense_hunt.json").read_text(encoding="utf-8"))

for name, hits in data["ranges"].items():
    strong = [h for h in hits if h["name"] >= 0.45 or h["death"] >= 0.4 or h["vic"] >= 0.4]
    print(f"\n=== {name} strong={len(strong)} ===")
    for h in strong:
        flag = []
        if h["name"] >= 0.45:
            flag.append("NAME")
        if h["death"] >= 0.4:
            flag.append("DEATH")
        if h["vic"] >= 0.4:
            flag.append("VIC")
        print(
            f"  t={h['t']:7.1f} name={h['name']:.2f} bar={h['bar']:.2f} "
            f"fill={h['fill']:.2f} d={h['death']:.2f} v={h['vic']:.2f} {' '.join(flag)}"
        )

print("\nSaved JPGs:")
for p in sorted(OUT.glob("*.jpg")):
    print(" ", p.name)
