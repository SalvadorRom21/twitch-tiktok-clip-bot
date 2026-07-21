"""YouTube Shorts upload via Google OAuth + Data API v3."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from twitch_tiktok_bot.config import AppConfig

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

_auth_lock = threading.Lock()
_auth_jobs: dict[str, dict[str, Any]] = {}
_auth_thread: threading.Thread | None = None


class YouTubeNotConfiguredError(RuntimeError):
    pass


def _secrets_path(config: AppConfig) -> Path:
    rel = config.publish.youtube.client_secrets_file
    return config.resolve_path(rel)


def _token_path(config: AppConfig) -> Path:
    return config.resolve_path(config.publish.youtube.token_file)


def _deps_ok() -> bool:
    try:
        import google.oauth2.credentials  # noqa: F401
        import google_auth_oauthlib.flow  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
    except ImportError:
        return False
    return True


def _token_valid(config: AppConfig) -> bool:
    token_path = _token_path(config)
    if not token_path.exists():
        return False
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        return False
    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.valid:
            return True
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def youtube_status(config: AppConfig) -> dict[str, Any]:
    secrets = _secrets_path(config)
    token = _token_path(config)
    deps = _deps_ok()
    configured = secrets.exists()
    authorized = _token_valid(config) if configured and deps else False
    auth_busy = _auth_thread is not None and _auth_thread.is_alive()
    if not deps:
        hint = (
            "Install YouTube deps: pip install google-api-python-client "
            "google-auth-oauthlib google-auth-httplib2"
        )
    elif not configured:
        hint = "Upload your Google OAuth Desktop client JSON (step 1 below)."
    elif not authorized:
        hint = "Client secrets ready - click Connect Google to finish browser login."
    else:
        hint = "YouTube connected - Shorts upload is ready."
    return {
        "configured": configured,
        "authorized": authorized,
        "deps_ok": deps,
        "auth_in_progress": auth_busy,
        "token_exists": token.exists(),
        "client_secrets_file": str(secrets),
        "token_file": str(token),
        "privacy": config.publish.youtube.privacy_status,
        "category_id": config.publish.youtube.category_id,
        "hint": hint,
        "console_url": "https://console.cloud.google.com/apis/credentials",
        "api_enable_url": "https://console.cloud.google.com/apis/library/youtube.googleapis.com",
        "consent_url": "https://console.cloud.google.com/auth/audience",
        "project_id": _project_id(secrets),
    }


def _project_id(secrets: Path) -> str:
    if not secrets.exists():
        return ""
    try:
        data = json.loads(secrets.read_text(encoding="utf-8"))
        block = data.get("installed") or data.get("web") or {}
        return str(block.get("project_id") or data.get("project_id") or "")
    except Exception:  # noqa: BLE001
        return ""


def validate_client_secrets_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept Google Desktop or Web OAuth client JSON."""
    if not isinstance(payload, dict):
        raise ValueError("Client secrets must be a JSON object")
    block = payload.get("installed") or payload.get("web")
    if not isinstance(block, dict):
        raise ValueError(
            "Expected Google OAuth client JSON with an 'installed' (Desktop) or 'web' key"
        )
    client_id = str(block.get("client_id") or "").strip()
    if not client_id:
        raise ValueError("client_id missing from OAuth client JSON")
    # Prefer Desktop shape for InstalledAppFlow.
    if "installed" not in payload and "web" in payload:
        payload = {"installed": dict(block)}
    return payload


def save_client_secrets(config: AppConfig, payload: dict[str, Any] | str | bytes) -> Path:
    if isinstance(payload, (str, bytes)):
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
    else:
        raw = payload
    cleaned = validate_client_secrets_payload(raw)
    path = _secrets_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return path


def disconnect_youtube(config: AppConfig) -> dict[str, Any]:
    token = _token_path(config)
    removed = False
    if token.exists():
        token.unlink()
        removed = True
    return {"disconnected": removed, **youtube_status(config)}


def authorize_youtube(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    """Run Google OAuth in this process (opens a local browser)."""
    if not _deps_ok():
        raise YouTubeNotConfiguredError(
            "Install YouTube deps: pip install google-api-python-client "
            "google-auth-oauthlib google-auth-httplib2"
        )
    secrets = _secrets_path(config)
    if not secrets.exists():
        raise YouTubeNotConfiguredError(
            f"Missing YouTube OAuth client secrets at {secrets}. "
            "Upload the Desktop client JSON from Google Cloud first."
        )
    if not force and _token_valid(config):
        return {"authorized": True, "already": True, **youtube_status(config)}

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True, prompt="consent")
    token_path = _token_path(config)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return {"authorized": True, "already": False, **youtube_status(config)}


def start_youtube_authorize(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    """Kick off browser OAuth on a background thread; poll get_authorize_job."""
    global _auth_thread
    with _auth_lock:
        if _auth_thread is not None and _auth_thread.is_alive():
            for job_id, job in _auth_jobs.items():
                if job.get("status") == "running":
                    return {"job_id": job_id, "status": "running", "already_running": True}
        job_id = f"yt_auth_{uuid.uuid4().hex[:10]}"
        _auth_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "message": "Waiting for Google login in your browser…",
            "error": "",
        }

        def _run() -> None:
            try:
                result = authorize_youtube(config, force=force)
                with _auth_lock:
                    _auth_jobs[job_id] = {
                        "job_id": job_id,
                        "status": "done",
                        "message": "YouTube connected.",
                        "error": "",
                        "result": result,
                    }
            except Exception as exc:  # noqa: BLE001
                with _auth_lock:
                    _auth_jobs[job_id] = {
                        "job_id": job_id,
                        "status": "failed",
                        "message": "Authorization failed",
                        "error": str(exc),
                    }

        _auth_thread = threading.Thread(target=_run, daemon=True)
        _auth_thread.start()
        return {"job_id": job_id, "status": "running", "already_running": False}


def get_authorize_job(job_id: str) -> dict[str, Any] | None:
    with _auth_lock:
        job = _auth_jobs.get(job_id)
        return dict(job) if job else None


def _build_youtube_service(config: AppConfig):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise YouTubeNotConfiguredError(
            "Install YouTube deps: pip install google-api-python-client "
            "google-auth-oauthlib google-auth-httplib2"
        ) from exc

    secrets = _secrets_path(config)
    if not secrets.exists():
        raise YouTubeNotConfiguredError(
            f"Missing YouTube OAuth client secrets at {secrets}. "
            "Create an OAuth Desktop client in Google Cloud Console "
            "(YouTube Data API v3), download JSON, and save it on the Auth page."
        )

    token_path = _token_path(config)
    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds)


def upload_short(
    config: AppConfig,
    *,
    video_path: Path,
    title: str,
    description: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Upload a vertical MP4 as a YouTube Short. Returns API response snippet."""
    from googleapiclient.http import MediaFileUpload

    if not video_path.exists():
        raise FileNotFoundError(f"Video missing: {video_path}")

    youtube = _build_youtube_service(config)
    ycfg = config.publish.youtube
    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": (tags or [])[:15],
            "categoryId": str(ycfg.category_id),
        },
        "status": {
            "privacyStatus": ycfg.privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    video_id = response.get("id", "")
    return {
        "video_id": video_id,
        "url": f"https://youtube.com/shorts/{video_id}" if video_id else "",
        "privacy": ycfg.privacy_status,
        "title": title,
        "raw": {
            "id": video_id,
            "snippet": response.get("snippet", {}),
            "status": response.get("status", {}),
        },
    }


def load_pack_youtube_texts(pack_dir: Path) -> tuple[str, str]:
    yt = pack_dir / "youtube"
    title = (yt / "title.txt").read_text(encoding="utf-8").strip() if (yt / "title.txt").exists() else ""
    desc = (
        (yt / "description.txt").read_text(encoding="utf-8")
        if (yt / "description.txt").exists()
        else ""
    )
    return title, desc


def save_upload_record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
