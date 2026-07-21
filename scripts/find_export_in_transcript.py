import re
from pathlib import Path

transcript = Path(
    r"C:\Users\Salvador Romero\.cursor\projects\c-Users-Salvador-Romero-Documents-twitch-tiktok-clip-bot"
    r"\agent-transcripts\e94afd13-b153-467d-bd3c-45cb3d0c2aca\e94afd13-b153-467d-bd3c-45cb3d0c2aca.jsonl"
)
text = transcript.read_text(encoding="utf-8", errors="ignore")
for pat in ['"example_count": 49', "5ed64cfc3f", "exported_at"]:
    idx = text.find(pat)
    print(pat, idx)
