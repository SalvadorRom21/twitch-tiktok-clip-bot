"""Re-analyze cached boss scan JPEGs with updated HUD detectors."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "ebd", ROOT / "twitch_tiktok_bot" / "labels" / "elden_boss_detect.py"
)
ebd = importlib.util.module_from_spec(spec)
sys.modules["ebd"] = ebd
assert spec.loader is not None
spec.loader.exec_module(ebd)

work_dir = ROOT / "data" / "be951f9be832"
scan_path = work_dir / "boss_scan.json"
frames_dir = work_dir / "boss_scan_frames"
data = json.loads(scan_path.read_text(encoding="utf-8"))
interval_sec = 2.0
coarse: list[ebd.FrameSignals] = []
idx = 0
while True:
    frame_path = frames_dir / f"frame_{idx:05d}.jpg"
    if not frame_path.exists():
        break
    img = cv2.imread(str(frame_path))
    if img is not None:
        coarse.append(ebd.analyze_frame(img, idx * interval_sec))
    idx += 1
if not coarse:
    raise SystemExit(f"No scan frames in {frames_dir}")

tuning_path = ROOT / "data" / "reference" / "elden_ring_tuning.json"
tuning = ebd.EldenTuning.from_dict(
    json.loads(tuning_path.read_text(encoding="utf-8"))["tuning"]
)
fresh = ebd.EldenTuning.defaults()
tuning.hud_bridge_sec = fresh.hud_bridge_sec
tuning.victory_hold_sec = fresh.victory_hold_sec
tuning.max_post_victory_sec = fresh.max_post_victory_sec
tuning.min_bar_when_name_high = fresh.min_bar_when_name_high
tuning.start_bar_lookback_sec = fresh.start_bar_lookback_sec
tuning.start_bar_min_score = fresh.start_bar_min_score

video_path = work_dir / f"{data['vod_id']}.mp4"
probe_dir = work_dir / "edge_probes"
probe_dir.mkdir(exist_ok=True)

# Probes refine clip start/end only — clustering stays on 2s coarse samples.
refined = list(coarse)
refined = ebd.supplement_start_probes(refined, video_path, tuning, probe_dir=probe_dir)
refined = ebd.supplement_end_probes(refined, video_path, tuning, probe_dir=probe_dir)

result = ebd.rescan_from_signals(
    coarse,
    data["duration_sec"],
    data["vod_id"],
    tuning=tuning,
    refine_signals=refined,
    data_dir=ROOT / "data",
)
ebd.save_scan_result(work_dir, result)
payload = json.loads(scan_path.read_text(encoding="utf-8"))
print("candidates", len(payload["candidates"]))
for c in sorted(payload["candidates"], key=lambda x: x["start_sec"]):
    dur = int(c["end_sec"] - c["start_sec"])
    print(int(c["start_sec"]), int(c["end_sec"]), dur, c["reason"][:55])
