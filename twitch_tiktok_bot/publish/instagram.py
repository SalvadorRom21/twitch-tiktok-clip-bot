"""Instagram publishing OAuth (Instagram API with Instagram Login)."""

from __future__ import annotations

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

AUTH_URL = "https://www.instagram.com/oauth/authorize"
TOKEN_URL = "https://api.instagram.com/oauth/access_token"
LONG_LIVED_URL = "https://graph.instagram.com/access_token"
ME_URL = "https://graph.instagram.com/v21.0/me"
SCOPES = [
    "instagram_business_basic",
    "instagram_business_content_publish",
]

_auth_lock = threading.Lock()
_auth_jobs: dict[str, dict[str, Any]] = {}
_auth_thread: threading.Thread | None = None


class InstagramNotConfiguredError(RuntimeError):
    pass


def _secrets_path(config: AppConfig) -> Path:
    return config.resolve_path(config.publish.instagram.client_secrets_file)


def _token_path(config: AppConfig) -> Path:
    return config.resolve_path(config.publish.instagram.token_file)


def _redirect_uri(config: AppConfig) -> str:
    port = int(config.publish.instagram.oauth_port or 8765)
    return f"http://127.0.0.1:{port}/callback/"


def _load_secrets(config: AppConfig) -> dict[str, str]:
    path = _secrets_path(config)
    if not path.exists():
        raise InstagramNotConfiguredError(
            f"Missing Instagram app secrets at {path}. Save app_id + app_secret on the Auth page."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    app_id = str(data.get("app_id") or data.get("client_id") or "").strip()
    app_secret = str(data.get("app_secret") or data.get("client_secret") or "").strip()
    if not app_id or not app_secret:
        raise InstagramNotConfiguredError("Instagram secrets JSON needs app_id and app_secret")
    return {"app_id": app_id, "app_secret": app_secret}


def validate_client_secrets_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Instagram secrets must be a JSON object")
    app_id = str(payload.get("app_id") or payload.get("client_id") or "").strip()
    app_secret = str(payload.get("app_secret") or payload.get("client_secret") or "").strip()
    if not app_id or not app_secret:
        raise ValueError('Expected JSON like {"app_id":"...","app_secret":"..."}')
    return {"app_id": app_id, "app_secret": app_secret}


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
        return bool(data.get("access_token"))
    except Exception:  # noqa: BLE001
        return False


def instagram_status(config: AppConfig) -> dict[str, Any]:
    secrets_path = _secrets_path(config)
    token = _token_path(config)
    configured = secrets_path.exists()
    authorized = _token_valid(config) if configured else False
    auth_busy = _auth_thread is not None and _auth_thread.is_alive()
    redirect = _redirect_uri(config)
    if not configured:
        hint = "Save your Meta Instagram app_id + app_secret."
    elif not authorized:
        hint = f"Client secrets ready - register redirect {redirect} then Connect Instagram."
    else:
        hint = "Instagram connected - Reels publish can use this token once Meta approves scopes."
    return {
        "platform": "instagram",
        "configured": configured,
        "authorized": authorized,
        "deps_ok": True,
        "auth_in_progress": auth_busy,
        "token_exists": token.exists(),
        "client_secrets_file": str(secrets_path),
        "token_file": str(token),
        "scopes": list(SCOPES),
        "redirect_uri": redirect,
        "hint": hint,
        "portal_url": "https://developers.facebook.com/apps/",
        "docs_url": (
            "https://developers.facebook.com/docs/instagram-platform/"
            "instagram-api-with-instagram-login/content-publishing/"
        ),
        "login_docs_url": (
            "https://developers.facebook.com/docs/instagram-platform/"
            "instagram-api-with-instagram-login/business-login/"
        ),
        "mode": "oauth" if authorized else ("ready_to_connect" if configured else "needs_secrets"),
        "note": (
            "Export packs still work without API. Direct Reels post needs a Professional "
            "Instagram account + Meta app with Instagram Login."
        ),
    }


def disconnect_instagram(config: AppConfig) -> dict[str, Any]:
    token = _token_path(config)
    removed = False
    if token.exists():
        token.unlink()
        removed = True
    return {"disconnected": removed, **instagram_status(config)}


def authorize_instagram(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    if not force and _token_valid(config):
        return {"authorized": True, "already": True, **instagram_status(config)}

    creds = _load_secrets(config)
    redirect_uri = _redirect_uri(config)
    port = int(config.publish.instagram.oauth_port or 8765)
    state = secrets.token_urlsafe(24)
    query = urlencode(
        {
            "client_id": creds["app_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": ",".join(SCOPES),
            "state": state,
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
    # Instagram sometimes appends #_ to the code
    code = code.rstrip("#_")
    if not code:
        raise RuntimeError("No authorization code returned from Instagram")

    token_resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": creds["app_id"],
            "client_secret": creds["app_secret"],
            "grant_type": "authorization_code",
            "redirect_uri": used_uri,
            "code": code,
        },
        timeout=60,
    )
    payload = token_resp.json() if token_resp.content else {}
    if token_resp.status_code >= 400 or payload.get("error_type") or payload.get("error"):
        raise RuntimeError(
            f"Instagram token exchange failed: {payload.get('error_message') or payload or token_resp.text}"
        )

    short_token = ""
    user_id = ""
    if isinstance(payload.get("data"), list) and payload["data"]:
        short_token = str(payload["data"][0].get("access_token") or "")
        user_id = str(payload["data"][0].get("user_id") or "")
    else:
        short_token = str(payload.get("access_token") or "")
        user_id = str(payload.get("user_id") or "")
    if not short_token:
        raise RuntimeError(f"Instagram token response missing access_token: {payload}")

    long_token = short_token
    long_expires = None
    try:
        long_resp = requests.get(
            LONG_LIVED_URL,
            params={
                "grant_type": "ig_exchange_token",
                "client_secret": creds["app_secret"],
                "access_token": short_token,
            },
            timeout=60,
        )
        long_payload = long_resp.json() if long_resp.content else {}
        if long_resp.ok and long_payload.get("access_token"):
            long_token = str(long_payload["access_token"])
            long_expires = long_payload.get("expires_in")
    except Exception:  # noqa: BLE001
        pass

    username = ""
    try:
        me = requests.get(
            ME_URL,
            params={"fields": "id,username", "access_token": long_token},
            timeout=30,
        )
        if me.ok:
            me_data = me.json()
            username = str(me_data.get("username") or "")
            if not user_id:
                user_id = str(me_data.get("id") or "")
    except Exception:  # noqa: BLE001
        pass

    token_path = _token_path(config)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "access_token": long_token,
        "user_id": user_id,
        "username": username,
        "expires_in": long_expires,
        "redirect_uri": used_uri,
        "scopes": SCOPES,
    }
    token_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {
        "authorized": True,
        "already": False,
        "username": username,
        "user_id": user_id,
        **instagram_status(config),
    }


def start_instagram_authorize(config: AppConfig, *, force: bool = False) -> dict[str, Any]:
    global _auth_thread
    with _auth_lock:
        if _auth_thread is not None and _auth_thread.is_alive():
            for job_id, job in _auth_jobs.items():
                if job.get("status") == "running":
                    return {"job_id": job_id, "status": "running", "already_running": True}
        job_id = f"ig_auth_{uuid.uuid4().hex[:10]}"
        _auth_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "message": "Waiting for Instagram login in your browser…",
            "error": "",
        }

        def _run() -> None:
            try:
                result = authorize_instagram(config, force=force)
                with _auth_lock:
                    _auth_jobs[job_id] = {
                        "job_id": job_id,
                        "status": "done",
                        "message": "Instagram connected.",
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
