# Twitch → TikTok Clip Bot

Automatically turn Twitch clips into short-form vertical videos for TikTok. The bot **understands** each clip (speech, loud moments, scenes, face position) and applies light, entertaining edits: montage jump cuts, zooms on reactions, captions, and a hook line.

Works **without an LLM** using rule-based editing. Enable LLM mode for smarter cut choices and optional vision descriptions.

## What it does

```text
Twitch clip URL
    → download (yt-dlp)
    → analyze (Whisper + audio peaks + face detection + scene detection)
    → edit plan (montage segments, rules or LLM)
    → render 9:16 MP4 + caption.txt
    → preview in web UI → approve before upload
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

## Web preview UI

Start the local preview server to process clips, watch the render, edit captions, and mark clips approved or rejected before uploading to TikTok.

```bash
python main.py --web
```

Open **http://127.0.0.1:8080** in your browser.

The UI lets you:

- Submit a Twitch clip URL and track processing status
- Preview the rendered vertical video
- Edit and save the TikTok caption
- Approve or reject clips for upload

Change host/port in `config.yaml` under `web:` or pass `--host` / `--port`.

## Configuration

Edit `config.yaml` or create `config.local.yaml` to override settings.

| Section | Key settings |
|---------|----------------|
| `twitch` | API credentials, `max_clips` |
| `analysis` | Whisper model size, vision frame interval |
| `editing` | Target duration, montage segments, zoom count |
| `render` | 1080×1920, face-aware crop, ffmpeg path |
| `llm` | Enable LLM planning, model, base URL |
| `web` | Preview UI host and port |

### Montage mode

By default the bot builds a **multi-segment montage** — it finds 2–4 highlight moments (reaction peaks, exciting lines) and stitches them into one TikTok short.

```yaml
editing:
  montage_enabled: true
  max_montage_segments: 4
  min_segment_sec: 4
  max_segment_sec: 12
```

Set `montage_enabled: false` for a single continuous highlight window.

### Face-aware crop

OpenCV detects your face cam and centers the 9:16 crop on it instead of blind center crop.

```yaml
render:
  face_crop_enabled: true
  face_sample_count: 8
```

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
  analyze/         # Whisper, audio peaks, face detection, vision
  plan/            # LLM + rule-based edit planning (montage)
  render/          # FFmpeg vertical render + face crop
  web/             # FastAPI preview UI
  pipeline.py      # End-to-end orchestration
main.py            # CLI entrypoint
config.yaml
```

## How “understanding” works

1. **Whisper** transcribes speech with timestamps.
2. **Audio analysis** finds loud peaks (reactions) and silence (cut opportunities).
3. **Face detection** locates the streamer for smart vertical cropping.
4. **Scene detection** spots hard visual cuts.
5. **Vision** (optional) samples frames and describes what’s on screen.
6. **Edit planner** picks montage segments or one highlight window, adds zooms, hook text, and captions.

## Customization ideas

- Add meme SFX in `assets/sfx/` and extend effect types in `plan/rules.py`
- Tune `editing.peak_percentile` if zooms fire too often or too rarely
- Swap Whisper model: `tiny` (fast) → `small` (better accuracy)
- Hook up TikTok Content Posting API for auto-upload (not included yet)

## Limitations

- Face crop uses OpenCV Haar cascades — works best with a clear face cam
- Montage segments are stitched with hard cuts (no crossfade yet)
- LLM vision uses up to 8 sampled frames to control API cost
- Read Twitch and TikTok terms before automating downloads/uploads at scale

## License

MIT — use and modify freely.
