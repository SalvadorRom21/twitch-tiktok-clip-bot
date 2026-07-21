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

## Recreate on GitHub (if the remote repo was deleted)

From this project folder, after [installing GitHub CLI](https://cli.github.com/) and running `gh auth login`:

```bash
chmod +x scripts/recreate_github_repo.sh
./scripts/recreate_github_repo.sh
```

That creates `SalvadorRom21/twitch-tiktok-clip-bot` and pushes `main`. Override names if needed:

```bash
GITHUB_OWNER=YourUsername GITHUB_REPO=your-repo-name ./scripts/recreate_github_repo.sh
```

**Manual option:** create an empty repo on GitHub, then:

```bash
git remote set-url origin https://github.com/YOUR_USER/twitch-tiktok-clip-bot.git
git push -u origin main
```

## Windows one-command setup

Open **PowerShell** and paste:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
irm https://raw.githubusercontent.com/SalvadorRom21/twitch-tiktok-clip-bot/main/setup-windows.ps1 | iex
```

This installs the project to:

```text
C:\Users\YOUR_USERNAME\Documents\twitch-tiktok-clip-bot
```

Then add Twitch credentials and run:

```powershell
notepad config.local.yaml
python main.py --web
```

## Quick start

```bash
# Clone and enter the repo
git clone https://github.com/SalvadorRom21/twitch-tiktok-clip-bot.git
cd twitch-tiktok-clip-bot

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add Twitch credentials (pick one method)

# Method A — YAML file (recommended)
cp config.local.yaml.example config.local.yaml
# Edit config.local.yaml and paste client_id, client_secret, broadcaster_id

# Method B — .env file
cp .env.example .env
# Edit .env and paste TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, TWITCH_BROADCASTER_ID

# Process one clip by URL
python main.py --clip-url "https://clips.twitch.tv/YourClipSlug"

# Process a full stream VOD (generates multiple TikTok shorts)
python main.py --vod-url "https://www.twitch.tv/videos/1234567890"

# Auto-detect clip vs VOD
python main.py --media-url "https://www.twitch.tv/videos/1234567890"

# Fetch recent stream VODs from your channel
python main.py --fetch-vods --max-shorts 3
```

Output lands in `output/`:

- Clips: `{clip_id}_tiktok.mp4` + caption `.txt`
- VODs: `{vod_id}_short_01.mp4`, `{vod_id}_short_02.mp4`, ... (one per highlight found)

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

### Full VOD / stream POV mode

Point the bot at an entire stream recording instead of a short clip. It scans the full VOD in chunks (audio reactions + speech), finds the best highlight moments, and renders **multiple TikTok shorts** automatically.

```yaml
vod:
  max_shorts_per_vod: 5      # how many TikToks per stream
  min_short_gap_sec: 120     # highlights must be 2+ min apart
  max_download_sec: 0        # 0 = full VOD; set e.g. 3600 to test first hour only
```

```bash
python main.py --vod-url "https://www.twitch.tv/videos/YOUR_VOD_ID" --max-shorts 5
python main.py --fetch-vods
```

Long VODs take longer (chunked Whisper transcription). Use a GPU or `whisper_model: tiny` for faster runs.

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

### LLM edit plans (recommended: Cursor — no OpenAI key)

The rule-based editor is fast but basic. For smarter montage cuts and hook text, use a **Cursor agent** as the editor:

1. Get a key at [cursor.com/dashboard/integrations](https://cursor.com/dashboard/integrations)
2. Add to `.env`:

```env
CURSOR_API_KEY=your_key_here
```

3. Enable in `config.local.yaml`:

```yaml
llm:
  enabled: true
  provider: cursor
  cursor_model: composer-2.5

analysis:
  whisper_model: small
  whisper_device: cuda        # NVIDIA GPU (e.g. RTX 4090)
  whisper_compute_type: float16
```

Re-run a clip — you should see `[plan] using Cursor edit plan` in the logs.

**OpenAI alternative:** set `provider: openai`, `OPENAI_API_KEY` in `.env`, and optionally `base_url` for Groq/etc.

## Project layout

```text
twitch_tiktok_bot/
  ingest/          # Twitch API + yt-dlp download
  analyze/         # Whisper, audio peaks, face detection, vision
  plan/            # LLM + rule-based edit planning (montage)
  render/          # FFmpeg vertical render + face crop
  publish/         # Trim, export packs (TT/IG/YT), YouTube Shorts upload
  web/             # FastAPI preview + trainer UI
  pipeline.py      # End-to-end orchestration
main.py            # CLI entrypoint
config.yaml
```

## Publish desk (edit → export → YouTube)

1. Process a Twitch clip **or** in Elden trainer click **Send to publish** on a fight.
2. Open `/` preview → select the ready short.
3. Mark **In/Out** on the player → **Re-trim clip** → edit caption.
4. **Export IG / YT / TT pack** → files land in `output/publish/{id}/` (one folder per platform with video + caption).
5. TikTok / Instagram: open the pack folder and upload from your phone.
6. YouTube Shorts (optional auto-upload):
   - Enable **YouTube Data API v3** in Google Cloud
   - Create an OAuth **Desktop** client, save JSON as `secrets/youtube_client_secret.json`
   - `pip install google-api-python-client google-auth-oauthlib google-auth-httplib2`
   - Click **Upload YouTube Short** (browser login once; default privacy is `private`)

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
- TikTok / Instagram Content APIs later — today use export packs + mobile apps
- YouTube Shorts auto-upload is supported via `publish/` (see Publish desk above)

## Limitations

- Face crop uses OpenCV Haar cascades — works best with a clear face cam
- Montage segments are stitched with hard cuts (no crossfade yet)
- LLM vision uses up to 8 sampled frames to control API cost
- Read Twitch and TikTok terms before automating downloads/uploads at scale

## License

MIT — use and modify freely.
