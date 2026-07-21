"""Learn from labeled examples and re-scan VODs using cached frame signals."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_load_module("twitch_tiktok_bot.labels.elden_boss_detect", "twitch_tiktok_bot/labels/elden_boss_detect.py")
supervised = _load_module(
    "twitch_tiktok_bot.labels.supervised",
    "twitch_tiktok_bot/labels/supervised.py",
)
retune_all_vods = supervised.retune_all_vods


def main() -> None:
    data_dir = ROOT / "data"
    reports = retune_all_vods(data_dir, clear_reviews=False)
    tuning_path = data_dir / "reference" / "elden_ring_tuning.json"
    print(json.dumps({"reports": reports, "tuning_path": str(tuning_path)}, indent=2))
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        for rule in tuning.get("rules", []):
            print(f"  - {rule}")


if __name__ == "__main__":
    main()
