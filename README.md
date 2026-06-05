# Twitch → TikTok Clip Bot

Automatically turn Twitch clips into short-form vertical videos for TikTok. The bot **understands** each clip (speech, loud moments, scenes) and applies light, entertaining edits: jump cuts on dead air, zooms on reactions, captions, and a hook line.

Works **without an LLM** using rule-based editing. Enable LLM mode for smarter cut choices and optional vision descriptions.

## What it does

```text
Twitch clip URL
    → download (yt-dlp)
    → analyze (Whisper + audio peaks + scene detection)
    → edit plan (rules or LLM)
    → render 9:16 MP4 + caption.txt
```

## Requirements

- **Python 3.10+**
- **FFmpeg** on your PATH (`ffmpeg -version`)
- **yt-dlp** (installed via pip)

Optional:

- **Twitch API credentials** — for `--fetch-clips`
- **OpenAI-compatible API key** — for LLM edit plans and vision frame descriptions
- **NVIDIA GPU** — set `analysis.whisper_device: cuda` in config for faster transcription

## Quick start

```bash
# Clone and enter the repo
cd twitch-tiktok-clip-bot

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env template (optional)
cp .env.example .env

# Process one clip by URL
python main.py --clip-url "https://clips.twitch.tv/YourClipSlug"
```

Output lands in `output/`:

- `{clip_id}_tiktok.mp4` — vertical short ready to upload
- `{clip_id}_tiktok.txt` — suggested caption + hashtags

Intermediate files (analysis JSON, edit plan, captions) are saved under `data/{clip_id}/`.

## Configuration

Edit `config.yaml` or create `config.local.yaml` to override settings.

| Section | Key settings |
|---------|----------------|
| `twitch` | API credentials, `max_clips` |
| `analysis` | Whisper model size, vision frame interval |
| `editing` | Target duration, zoom count, silence threshold |
| `render` | 1080×1920, fps, custom ffmpeg path |
| `llm` | Enable LLM planning, model, base URL |

### Twitch API setup

1. Create an app at [Twitch Developer Console](https://dev.twitch.tv/console/apps)
2. Set env vars or config:
   - `TWITCH_CLIENT_ID`
   - `TWITCH_CLIENT_SECRET`
   - `TWITCH_BROADCASTER_ID` (your numeric user ID)

Then fetch and process recent clips:

```bash
python main.py --fetch-clips
```

### LLM mode (optional)

Set in `config.local.yaml`:

```yaml
llm:
  enabled: true
  model: gpt-4o-mini
  # base_url: https://api.groq.com/openai/v1  # for Groq, etc.
```

And set `OPENAI_API_KEY` in `.env`. Without this, the bot uses built-in rule-based editing.

## Project layout

```text
twitch_tiktok_bot/
  ingest/          # Twitch API + yt-dlp download
  analyze/         # Whisper, audio peaks, vision frames
  plan/            # LLM + rule-based edit planning
  render/          # FFmpeg vertical render
  pipeline.py      # End-to-end orchestration
main.py            # CLI entrypoint
config.yaml
```

## How “understanding” works

1. **Whisper** transcribes speech with timestamps.
2. **Audio analysis** finds loud peaks (reactions) and silence (cut opportunities).
3. **Scene detection** spots hard visual cuts.
4. **Vision** (optional) samples frames and describes what’s on screen.
5. **Edit planner** picks the best ~30s window and adds zooms, hook text, and captions.

## Customization ideas

- Add meme SFX in `assets/sfx/` and extend effect types in `plan/rules.py`
- Tune `editing.peak_percentile` if zooms fire too often or too rarely
- Swap Whisper model: `tiny` (fast) → `small` (better accuracy)
- Hook up TikTok Content Posting API for auto-upload (not included in MVP)

## Limitations (MVP)

- One highlight window per clip (not multi-segment montage yet)
- Face-aware crop not implemented — uses center crop
- LLM vision uses up to 8 sampled frames to control API cost
- Read Twitch and TikTok terms before automating downloads/uploads at scale

## License

MIT — use and modify freely.
