"""TikTok Content Posting API OAuth (Login Kit Desktop + PKCE)."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.publish.oauth_local import run_local_oauth_callback

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"
SCOPES = ["user.info.basic", "video.upload", "video.publish"]

_auth_lock = threading.Lock()
_auth_jobs: dict[str, dict[str, Any]] = {}
_auth_thread: threading.Thread | None = None


class TikTokNotConfiguredError(RuntimeError):
    pass


def _secrets_path(config: AppConfig) -> Path:
    return config.resolve_path(config.publish.tiktok.client_secrets_file)


def _token_path(config: AppConfig) -> Path:
    return config.resolve_path(config.publish.tiktok.token_file)


def _load_secrets(config: AppConfig) -> dict[str, str]:
    path = _secrets_path(config)
    if not path.exists():
        raise TikTokNotConfiguredError(
            f"Missing TikTok client secrets at {path}. Save client_key + client_secret on the Auth page."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    key = str(data.get("client_key") or data.get("client_id") or "").strip()
    secret = str(data.get("client_secret") or "").strip()
    if not key or not secret:
        raise TikTokNotConfiguredError("TikTok secrets JSON needs client_key and client_secret")
    return {"client_key": key, "client_secret": secret}


def validate_client_secrets_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("TikTok secrets must be a JSON object")
    key = str(payload.get("client_key") or payload.get("client_id") or "").strip()
    secret = str(payload.get("client_secret") or "").strip()
    if not key or not secret:
        raise ValueError('Expected JSON like {"client_key":"...","client_secret":"..."}')
    return {"client_key": key, "client_secret": secret}


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


def _token_valid(config: AppConfig) -> bool:
    path = _token_path(config)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("access_token") or data.get("refresh_token"))
    except Exception:  # noqa: BLE001
        return False


def tiktok_status(config: AppConfig) -> dict[str, Any]:
    secrets_path = _secrets_path(config)
    token = _token_path(config)
    configured = secrets_path.exists()
    authorized = _token_valid(config) if configured else False
    auth_busy = _auth_thread is not None and _auth_thread.is_alive()
    if not configured:
        hint = "Save your TikTok client_key + client_secret (from developers.tiktok.com)."
    elif not authorized:
        hint = "Client secrets ready - click Connect TikTok (sandbox / approved scopes required)."
    else:
        hint = "TikTok connected - posting API can use this token once scopes are approved."
    return {
        "platform": "tiktok",
        "configured": configured,
        "authorized": authorized,
        "deps_ok": True,
        "auth_in_progress": auth_busy,
        "token_exists": token.exists(),
        "client_secrets_file": str(secrets_path),
        "token_file": str(token),
        "scopes": list(SCOPES),
        "hint": hint,
        "portal_url": "https://developers.tiktok.com/apps/",
        "docs_url": "https://developers.tiktok.com/doc/content-posting-api-get-started",
        "login_kit_url": "https://developers.tiktok.com/doc/login-kit-desktop/",
        "redirect_hint": "http://127.0.0.1:8766/callback/",
        "redirect_uri": f"http://127.0.0.1:{int(config.publish.tiktok.oauth_port or 8766)}/callback/",
        "mode": "oauth" if authorized else ("ready_to_connect" if configured else "needs_secrets"),
        "note": (
            "Export packs still work without API. Direct post needs Content Posting API + "
            "video.publish approval (sandbox first)."
        ),
    }


def disconnect_tiktok(config: AppConfig) -> dict[str, Any]:
    token = _token_path(config)
    removed = False
    if token.exists():
        token.unlink()
        removed = True
    return {"disconnected": removed, **tiktok_status(config)}


def _pkce_pair() -> tuple[str, str]:
    """TikTok Desktop Login Kit expects HEX SHA-256 code_challenge."""
    verifier = secrets.token_urlsafe(64)
    challenge = hashlib.sha256(verifier.encode("utf-8")).hexdigest()
    return verifier, challenge


def authorize_tiktok(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    if not force and _token_valid(config):
        return {"authorized": True, "already": True, **tiktok_status(config)}

    creds = _load_secrets(config)
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)
    port = int(config.publish.tiktok.oauth_port or 8766)
    redirect_uri = f"http://127.0.0.1:{port}/callback/"
    query = urlencode(
        {
            "client_key": creds["client_key"],
            "response_type": "code",
            "scope": ",".join(SCOPES),
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    auth_url = f"{AUTH_URL}?{query}"

    used_uri, params = run_local_oauth_callback(
        host="127.0.0.1",
        port=port,
        path="/callback/",
        before_wait=lambda _uri: webbrowser.open(auth_url),
    )
    if params.get("state") and params.get("state") != state:
        raise RuntimeError("OAuth state mismatch - try Connect again")
    code = str(params.get("code") or "").strip()
    if not code:
        raise RuntimeError("No authorization code returned from TikTok")

    token_resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_key": creds["client_key"],
            "client_secret": creds["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": used_uri,
            "code_verifier": verifier,
        },
        timeout=60,
    )
    payload = token_resp.json() if token_resp.content else {}
    if token_resp.status_code >= 400 or payload.get("error"):
        raise RuntimeError(
            f"TikTok token exchange failed: {payload.get('error_description') or payload or token_resp.text}"
        )

    token_path = _token_path(config)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "access_token": payload.get("access_token"),
        "refresh_token": payload.get("refresh_token"),
        "expires_in": payload.get("expires_in"),
        "refresh_expires_in": payload.get("refresh_expires_in"),
        "open_id": payload.get("open_id"),
        "scope": payload.get("scope"),
        "token_type": payload.get("token_type"),
        "redirect_uri": used_uri,
    }
    token_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"authorized": True, "already": False, "open_id": record.get("open_id"), **tiktok_status(config)}


def start_tiktok_authorize(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    global _auth_thread
    with _auth_lock:
        if _auth_thread is not None and _auth_thread.is_alive():
            for job_id, job in _auth_jobs.items():
                if job.get("status") == "running":
                    return {"job_id": job_id, "status": "running", "already_running": True}
        job_id = f"tt_auth_{uuid.uuid4().hex[:10]}"
        _auth_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "message": "Waiting for TikTok login in your browser…",
            "error": "",
        }

        def _run() -> None:
            try:
                result = authorize_tiktok(config, force=force)
                with _auth_lock:
                    _auth_jobs[job_id] = {
                        "job_id": job_id,
                        "status": "done",
                        "message": "TikTok connected.",
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
