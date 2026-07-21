"""FastAPI web app for previewing and approving TikTok shorts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twitch_tiktok_bot.config import AppConfig, apply_game_profile, load_config
from twitch_tiktok_bot.labels.fights import (
    FightLabelStore,
    find_cached_vods,
    load_fight_labels,
    new_fight_label,
    resolve_vod_video,
    save_fight_labels,
)
from twitch_tiktok_bot.labels.elden_feedback import (
    feedback_stats,
    load_feedback_store,
    record_feedback,
    save_guide,
)
from twitch_tiktok_bot.labels.supervised import (
    VERDICT_CORRECT,
    VERDICT_WRONG,
    WRONG_REASONS,
    delete_all_train_vods,
    delete_train_vod,
    load_supervised_store,
    reset_elden_training,
    retune_all_vods,
    review_for,
    save_supervised_store,
)
from twitch_tiktok_bot.labels.literacy import (
    CLIP_WORTHY_OPTIONS,
    EVENT_TYPES,
    LiteracyMoment,
    ensure_literacy_store,
    load_literacy_store,
    save_literacy_store,
    sync_literacy_to_reference,
)
from twitch_tiktok_bot.plan.game_profiles import GAME_PROFILE_OPTIONS
from twitch_tiktok_bot.status import ClipStatus, list_jobs, load_job
from twitch_tiktok_bot.web.jobs import set_status, start_job, update_caption
from twitch_tiktok_bot.publish import (
    create_job_from_vod_window,
    export_clip_pack,
    get_publish_status,
    start_publish_from_train,
    trim_clip_job,
    upload_clip_to_youtube,
)
from twitch_tiktok_bot.publish.youtube import (
    YouTubeNotConfiguredError,
    disconnect_youtube,
    get_authorize_job as get_youtube_authorize_job,
    save_client_secrets as save_youtube_secrets,
    start_youtube_authorize,
    youtube_status,
)
from twitch_tiktok_bot.publish.tiktok import (
    TikTokNotConfiguredError,
    disconnect_tiktok,
    get_authorize_job as get_tiktok_authorize_job,
    save_client_secrets as save_tiktok_secrets,
    start_tiktok_authorize,
    tiktok_status,
)
from twitch_tiktok_bot.publish.instagram import (
    InstagramNotConfiguredError,
    disconnect_instagram,
    get_authorize_job as get_instagram_authorize_job,
    save_client_secrets as save_instagram_secrets,
    start_instagram_authorize,
    instagram_status,
)

STATIC_DIR = Path(__file__).parent / "static"


class ProcessRequest(BaseModel):
    clip_url: str
    title: str = ""
    clip_id: str = ""
    game_profile: str = ""


class CaptionUpdate(BaseModel):
    caption: str


class TrimBody(BaseModel):
    start_sec: float
    end_sec: float
    caption: str | None = None


class ExportBody(BaseModel):
    caption: str | None = None


class YoutubeUploadBody(BaseModel):
    caption: str | None = None


class YoutubeSecretsBody(BaseModel):
    """Paste Google OAuth client JSON (Desktop preferred)."""

    json_text: str = ""


class YoutubeAuthorizeBody(BaseModel):
    force: bool = False


class PlatformSecretsBody(BaseModel):
    json_text: str = ""


class PlatformAuthorizeBody(BaseModel):
    force: bool = False


class PublishFromTrainBody(BaseModel):
    start_sec: float
    end_sec: float
    title: str = ""
    caption: str = ""


class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected|ready)$")


class FightLabelCreate(BaseModel):
    start_sec: float
    end_sec: float
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    quality: str = "good"
    use_for_clips: bool = True
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    notes: str = ""


class FightLabelUpdate(BaseModel):
    start_sec: float | None = None
    end_sec: float | None = None
    description: str | None = None
    tags: list[str] | None = None
    quality: str | None = None
    use_for_clips: bool | None = None
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    notes: str | None = None


class GuideUpdate(BaseModel):
    guide: str = ""


class LiteracyMomentCreate(BaseModel):
    timestamp_sec: float
    title: str = "Custom moment"
    bot_question: str = ""
    category: str = "unknown"


class LiteracyMomentUpdate(BaseModel):
    event_type: str | None = None
    what_happening: str | None = None
    visual_cues: str | None = None
    audio_cues: str | None = None
    clip_worthy: str | None = None
    teaches_bot: str | None = None
    answered: bool | None = None


class GeneralLiteracyUpdate(BaseModel):
    answer: str = ""


class CandidateReviewBody(BaseModel):
    verdict: str = Field(..., pattern="^(correct|wrong)$")
    wrong_reason: str = ""
    boss_type: str = ""
    notes: str = ""
    start_sec: float | None = None
    end_sec: float | None = None


class TimingBody(BaseModel):
    start_sec: float | None = None
    end_sec: float | None = None


class MissedFightBody(BaseModel):
    start_sec: float
    end_sec: float
    boss_type: str = ""
    notes: str = ""


class TrainResetBody(BaseModel):
    confirm: str = ""


class TrainGuideBody(BaseModel):
    guide: str = ""


class MlLabelBody(BaseModel):
    label: str
    path: str
    vod_id: str = ""
    time_sec: float | None = None


class MlTrainBody(BaseModel):
    epochs: int = Field(default=12, ge=1, le=50)
    min_labels: int = Field(default=20, ge=8, le=500)


class MericaPackBody(BaseModel):
    force: bool = False
    extract_video: bool = False


class ManualClipBody(BaseModel):
    start_sec: float
    end_sec: float
    kind: str = "you_died"
    notes: str = ""
    extract_video: bool = True
    expand_frames: bool = True


class ManualClipUpdateBody(BaseModel):
    start_sec: float | None = None
    end_sec: float | None = None
    kind: str | None = None
    notes: str | None = None
    extract_video: bool = True
    expand_frames: bool = True


def create_app(config: AppConfig | None = None) -> FastAPI:
    app = FastAPI(title="Twitch TikTok Clip Bot", version="0.2.0")
    app.state.config = config or load_config()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="UI not found")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/label", response_class=HTMLResponse)
    def label_page() -> HTMLResponse:
        label_path = STATIC_DIR / "label.html"
        if not label_path.exists():
            raise HTTPException(status_code=500, detail="Label UI not found")
        return HTMLResponse(label_path.read_text(encoding="utf-8"))

    @app.get("/literacy", response_class=HTMLResponse)
    def literacy_page() -> HTMLResponse:
        literacy_path = STATIC_DIR / "literacy.html"
        if not literacy_path.exists():
            raise HTTPException(status_code=500, detail="Literacy UI not found")
        return HTMLResponse(literacy_path.read_text(encoding="utf-8"))

    @app.get("/train", response_class=HTMLResponse)
    def train_page() -> HTMLResponse:
        train_path = STATIC_DIR / "train.html"
        if not train_path.exists():
            raise HTTPException(status_code=500, detail="Train UI not found")
        return HTMLResponse(train_path.read_text(encoding="utf-8"))

    @app.get("/auth", response_class=HTMLResponse)
    def auth_page() -> HTMLResponse:
        auth_path = STATIC_DIR / "auth.html"
        if not auth_path.exists():
            raise HTTPException(status_code=500, detail="Auth UI not found")
        return HTMLResponse(auth_path.read_text(encoding="utf-8"))

    @app.get("/api/game-profiles")
    def get_game_profiles() -> dict:
        cfg: AppConfig = app.state.config
        return {
            "options": GAME_PROFILE_OPTIONS,
            "default": cfg.editing.game_profile,
        }

    @app.get("/api/clips")
    def get_clips() -> list[dict]:
        cfg: AppConfig = app.state.config
        return [job.to_dict() for job in list_jobs(cfg)]

    @app.get("/api/clips/{clip_id}")
    def get_clip(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        job = load_job(cfg, clip_id)
        if not job:
            raise HTTPException(status_code=404, detail="Clip not found")
        return job.to_dict()

    @app.post("/api/process")
    def process_clip(body: ProcessRequest) -> dict:
        cfg: AppConfig = app.state.config
        if not body.clip_url.strip():
            raise HTTPException(status_code=400, detail="clip_url is required")
        job = start_job(
            cfg,
            clip_url=body.clip_url.strip(),
            title=body.title.strip(),
            clip_id=body.clip_id.strip() or None,
            game_profile=body.game_profile.strip() or None,
        )
        return job.to_dict()

    @app.get("/api/clips/{clip_id}/video")
    def get_video(clip_id: str) -> FileResponse:
        cfg: AppConfig = app.state.config
        job = load_job(cfg, clip_id)
        if not job or not job.output_video:
            raise HTTPException(status_code=404, detail="Video not found")
        video_path = Path(job.output_video)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file missing")
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/clips/{clip_id}/caption")
    def get_caption(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        job = load_job(cfg, clip_id)
        if not job:
            raise HTTPException(status_code=404, detail="Clip not found")
        caption_path = Path(job.caption_file) if job.caption_file else None
        text = ""
        if caption_path and caption_path.exists():
            text = caption_path.read_text(encoding="utf-8")
        return {"caption": text, "hook_text": job.hook_text, "hashtags": job.hashtags}

    @app.put("/api/clips/{clip_id}/caption")
    def put_caption(clip_id: str, body: CaptionUpdate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = update_caption(cfg, clip_id, body.caption)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/clips/{clip_id}/approve")
    def approve_clip(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = set_status(cfg, clip_id, ClipStatus.APPROVED)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/clips/{clip_id}/reject")
    def reject_clip(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = set_status(cfg, clip_id, ClipStatus.REJECTED)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/clips/{clip_id}/status")
    def update_status(clip_id: str, body: StatusUpdate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = set_status(cfg, clip_id, ClipStatus(body.status))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.get("/api/publish/youtube/status")
    def get_youtube_publish_status() -> dict:
        cfg: AppConfig = app.state.config
        return youtube_status(cfg)

    @app.get("/api/publish/status")
    def get_all_publish_auth_status() -> dict:
        cfg: AppConfig = app.state.config
        return {
            "youtube": youtube_status(cfg),
            "tiktok": tiktok_status(cfg),
            "instagram": instagram_status(cfg),
        }

    @app.post("/api/publish/youtube/secrets")
    async def upload_youtube_secrets(
        file: UploadFile | None = File(None),
        json_text: str = Form(""),
    ) -> dict:
        """Save Google OAuth client JSON via multipart file and/or pasted text."""
        cfg: AppConfig = app.state.config
        raw: str | bytes = ""
        if file is not None and file.filename:
            raw = await file.read()
        elif json_text.strip():
            raw = json_text
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide a client secrets JSON file or paste json_text",
            )
        try:
            path = save_youtube_secrets(cfg, raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status = youtube_status(cfg)
        return {"saved": True, "path": str(path), **status}

    @app.post("/api/publish/youtube/secrets/json")
    def save_youtube_secrets_json(body: YoutubeSecretsBody) -> dict:
        """Save Google OAuth client JSON from a JSON body (paste in UI)."""
        cfg: AppConfig = app.state.config
        if not body.json_text.strip():
            raise HTTPException(status_code=400, detail="json_text is empty")
        try:
            path = save_youtube_secrets(cfg, body.json_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status = youtube_status(cfg)
        return {"saved": True, "path": str(path), **status}

    @app.post("/api/publish/youtube/authorize")
    def authorize_youtube_account(body: YoutubeAuthorizeBody | None = None) -> dict:
        """Open a browser for Google OAuth (runs in background). Poll the job_id."""
        cfg: AppConfig = app.state.config
        force = body.force if body else False
        try:
            return start_youtube_authorize(cfg, force=force)
        except YouTubeNotConfiguredError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/publish/youtube/authorize/{job_id}")
    def authorize_youtube_job(job_id: str) -> dict:
        job = get_youtube_authorize_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Auth job not found")
        return job

    @app.delete("/api/publish/youtube/token")
    def disconnect_youtube_account() -> dict:
        cfg: AppConfig = app.state.config
        return disconnect_youtube(cfg)

    @app.get("/api/publish/tiktok/status")
    def get_tiktok_publish_status() -> dict:
        cfg: AppConfig = app.state.config
        return tiktok_status(cfg)

    @app.post("/api/publish/tiktok/secrets")
    async def upload_tiktok_secrets(
        file: UploadFile | None = File(None),
        json_text: str = Form(""),
    ) -> dict:
        cfg: AppConfig = app.state.config
        raw: str | bytes = ""
        if file is not None and file.filename:
            raw = await file.read()
        elif json_text.strip():
            raw = json_text
        else:
            raise HTTPException(status_code=400, detail="Provide TikTok secrets JSON file or text")
        try:
            path = save_tiktok_secrets(cfg, raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"saved": True, "path": str(path), **tiktok_status(cfg)}

    @app.post("/api/publish/tiktok/secrets/json")
    def save_tiktok_secrets_json(body: PlatformSecretsBody) -> dict:
        cfg: AppConfig = app.state.config
        if not body.json_text.strip():
            raise HTTPException(status_code=400, detail="json_text is empty")
        try:
            path = save_tiktok_secrets(cfg, body.json_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"saved": True, "path": str(path), **tiktok_status(cfg)}

    @app.post("/api/publish/tiktok/authorize")
    def authorize_tiktok_account(body: PlatformAuthorizeBody | None = None) -> dict:
        cfg: AppConfig = app.state.config
        force = body.force if body else False
        try:
            return start_tiktok_authorize(cfg, force=force)
        except TikTokNotConfiguredError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/publish/tiktok/authorize/{job_id}")
    def authorize_tiktok_job(job_id: str) -> dict:
        job = get_tiktok_authorize_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Auth job not found")
        return job

    @app.delete("/api/publish/tiktok/token")
    def disconnect_tiktok_account() -> dict:
        cfg: AppConfig = app.state.config
        return disconnect_tiktok(cfg)

    @app.get("/api/publish/instagram/status")
    def get_instagram_publish_status() -> dict:
        cfg: AppConfig = app.state.config
        return instagram_status(cfg)

    @app.post("/api/publish/instagram/secrets")
    async def upload_instagram_secrets(
        file: UploadFile | None = File(None),
        json_text: str = Form(""),
    ) -> dict:
        cfg: AppConfig = app.state.config
        raw: str | bytes = ""
        if file is not None and file.filename:
            raw = await file.read()
        elif json_text.strip():
            raw = json_text
        else:
            raise HTTPException(status_code=400, detail="Provide Instagram secrets JSON file or text")
        try:
            path = save_instagram_secrets(cfg, raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"saved": True, "path": str(path), **instagram_status(cfg)}

    @app.post("/api/publish/instagram/secrets/json")
    def save_instagram_secrets_json(body: PlatformSecretsBody) -> dict:
        cfg: AppConfig = app.state.config
        if not body.json_text.strip():
            raise HTTPException(status_code=400, detail="json_text is empty")
        try:
            path = save_instagram_secrets(cfg, body.json_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"saved": True, "path": str(path), **instagram_status(cfg)}

    @app.post("/api/publish/instagram/authorize")
    def authorize_instagram_account(body: PlatformAuthorizeBody | None = None) -> dict:
        cfg: AppConfig = app.state.config
        force = body.force if body else False
        try:
            return start_instagram_authorize(cfg, force=force)
        except InstagramNotConfiguredError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/publish/instagram/authorize/{job_id}")
    def authorize_instagram_job(job_id: str) -> dict:
        job = get_instagram_authorize_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Auth job not found")
        return job

    @app.delete("/api/publish/instagram/token")
    def disconnect_instagram_account() -> dict:
        cfg: AppConfig = app.state.config
        return disconnect_instagram(cfg)

    @app.get("/api/clips/{clip_id}/publish")
    def get_clip_publish(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            return get_publish_status(cfg, clip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/clips/{clip_id}/trim")
    def trim_clip(clip_id: str, body: TrimBody) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = trim_clip_job(
                cfg,
                clip_id,
                start_sec=body.start_sec,
                end_sec=body.end_sec,
                caption=body.caption,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/clips/{clip_id}/export")
    def export_clip(clip_id: str, body: ExportBody | None = None) -> dict:
        cfg: AppConfig = app.state.config
        caption = body.caption if body else None
        try:
            pack = export_clip_pack(cfg, clip_id, caption=caption)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return pack

    @app.post("/api/clips/{clip_id}/publish/youtube")
    def publish_youtube(clip_id: str, body: YoutubeUploadBody | None = None) -> dict:
        cfg: AppConfig = app.state.config
        caption = body.caption if body else None
        try:
            return upload_clip_to_youtube(cfg, clip_id, caption=caption)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except YouTubeNotConfiguredError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/train/{vod_id}/publish")
    def publish_from_train(vod_id: str, body: PublishFromTrainBody) -> dict:
        """Start async extract of a VOD window into the publish desk."""
        cfg: AppConfig = app.state.config
        if body.end_sec <= body.start_sec:
            raise HTTPException(status_code=400, detail="end_sec must be > start_sec")
        # Confirm VOD exists before queuing.
        try:
            resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        job_id = start_publish_from_train(
            cfg,
            vod_id=vod_id,
            start_sec=body.start_sec,
            end_sec=body.end_sec,
            title=body.title.strip(),
            caption=body.caption.strip(),
        )
        return {
            "job_id": job_id,
            "status": "started",
            "vod_id": vod_id,
            "start_sec": body.start_sec,
            "end_sec": body.end_sec,
            "clip_duration_sec": max(0.0, body.end_sec - body.start_sec),
        }

    @app.get("/api/label/vods")
    def list_label_vods() -> list[dict]:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        return find_cached_vods(data_dir)

    @app.get("/api/label/{vod_id}")
    def get_label_store(vod_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        return {
            **store.to_dict(),
            "video_url": f"/api/label/{vod_id}/video",
            "video_name": video_path.name,
        }

    @app.get("/api/label/{vod_id}/video")
    def get_label_video(vod_id: str) -> FileResponse:
        cfg: AppConfig = app.state.config
        try:
            _work_dir, video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"},
        )

    @app.put("/api/label/{vod_id}/guide")
    def put_label_guide(vod_id: str, body: GuideUpdate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        store.guide = body.guide.strip()
        save_fight_labels(work_dir, store)
        return store.to_dict()

    @app.post("/api/label/{vod_id}/fights")
    def add_fight_label(vod_id: str, body: FightLabelCreate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        fight = new_fight_label(
            start_sec=body.start_sec,
            end_sec=body.end_sec,
            description=body.description,
            tags=body.tags,
            quality=body.quality,
            use_for_clips=body.use_for_clips,
            notes=body.notes,
        )
        if body.clip_start_sec is not None:
            fight.clip_start_sec = round(body.clip_start_sec, 2)
        if body.clip_end_sec is not None:
            fight.clip_end_sec = round(body.clip_end_sec, 2)
        store.fights.append(fight)
        store.fights.sort(key=lambda item: item.start_sec)
        save_fight_labels(work_dir, store)
        return fight.to_dict()

    @app.put("/api/label/{vod_id}/fights/{fight_id}")
    def update_fight_label(
        vod_id: str, fight_id: str, body: FightLabelUpdate
    ) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        fight = next((item for item in store.fights if item.id == fight_id), None)
        if fight is None:
            raise HTTPException(status_code=404, detail="Fight label not found")
        updates = body.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(fight, key, value)
        if fight.end_sec < fight.start_sec:
            fight.start_sec, fight.end_sec = fight.end_sec, fight.start_sec
        store.fights.sort(key=lambda item: item.start_sec)
        save_fight_labels(work_dir, store)
        return fight.to_dict()

    @app.delete("/api/label/{vod_id}/fights/{fight_id}")
    def delete_fight_label(vod_id: str, fight_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        before = len(store.fights)
        store.fights = [item for item in store.fights if item.id != fight_id]
        if len(store.fights) == before:
            raise HTTPException(status_code=404, detail="Fight label not found")
        save_fight_labels(work_dir, store)
        return store.to_dict()

    @app.get("/api/literacy")
    def literacy_api_index() -> dict:
        return {
            "ui": "/literacy",
            "vods": "/api/literacy/vods",
            "store": "/api/literacy/{vod_id}",
            "hint": "Start server: python main.py --literacy 2788855626",
        }

    @app.get("/api/literacy/vods")
    def list_literacy_vods() -> list[dict]:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        vods = find_cached_vods(data_dir)
        for vod in vods:
            work_dir = data_dir / vod["vod_id"]
            literacy = load_literacy_store(work_dir)
            vod["moment_count"] = len(literacy.moments) if literacy else 0
        return vods

    @app.get("/api/literacy/{vod_id}")
    def get_literacy_store(vod_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        save_literacy_store(work_dir, store)
        answered = sum(1 for moment in store.moments if moment.answered)
        return {
            **store.to_dict(),
            "video_url": f"/api/label/{vod_id}/video",
            "event_types": EVENT_TYPES,
            "clip_worthy_options": CLIP_WORTHY_OPTIONS,
            "progress": {
                "answered": answered,
                "total": len(store.moments),
            },
        }

    @app.post("/api/literacy/{vod_id}/moments")
    def add_literacy_moment(vod_id: str, body: LiteracyMomentCreate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        moment = LiteracyMoment(
            id=uuid.uuid4().hex[:10],
            timestamp_sec=round(body.timestamp_sec, 2),
            title=body.title.strip() or "Custom moment",
            bot_question=body.bot_question.strip()
            or "What is happening here?",
            category=body.category,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        store.moments.append(moment)
        store.moments.sort(key=lambda item: item.timestamp_sec)
        save_literacy_store(work_dir, store)
        return moment.to_dict()

    @app.put("/api/literacy/{vod_id}/moments/{moment_id}")
    def update_literacy_moment(
        vod_id: str, moment_id: str, body: LiteracyMomentUpdate
    ) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        moment = next(
            (item for item in store.moments if item.id == moment_id), None
        )
        if moment is None:
            raise HTTPException(status_code=404, detail="Moment not found")
        for key, value in body.model_dump(exclude_unset=True).items():
            setattr(moment, key, value)
        save_literacy_store(work_dir, store)
        return moment.to_dict()

    @app.put("/api/literacy/{vod_id}/general/{question_id}")
    def update_general_literacy(
        vod_id: str, question_id: str, body: GeneralLiteracyUpdate
    ) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        item = next((g for g in store.general if g.id == question_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="Question not found")
        item.answer = body.answer.strip()
        save_literacy_store(work_dir, store)
        return item.to_dict()

    @app.post("/api/literacy/{vod_id}/sync")
    def sync_literacy(vod_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_literacy_store(work_dir)
        if store is None:
            raise HTTPException(status_code=404, detail="No literacy labels yet")
        out_path = sync_literacy_to_reference(
            store,
            cfg.resolve_path("data/reference/apex_game_literacy.yaml"),
            cfg.project_root,
        )
        answered = sum(1 for moment in store.moments if moment.answered)
        return {
            "synced_to": str(out_path),
            "moments_answered": answered,
            "moments_total": len(store.moments),
        }

    # --- Supervised Elden Ring training ---

    @app.get("/api/train/meta")
    def train_meta() -> dict:
        from twitch_tiktok_bot.labels.elden_ml.config import CLASS_LABELS, load_ml_config
        from twitch_tiktok_bot.labels.elden_ml.dataset import label_counts
        from twitch_tiktok_bot.labels.elden_ml.infer import model_is_ready

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        ml_cfg = load_ml_config(data_dir)
        return {
            "wrong_reasons": WRONG_REASONS,
            "mode": "ml",
            "classes": CLASS_LABELS,
            "model_ready": model_is_ready(data_dir),
            "label_counts": label_counts(data_dir),
            "ml": ml_cfg.to_dict(),
        }

    @app.get("/api/train/guide")
    def get_train_guide() -> dict:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        store = load_feedback_store(data_dir)
        return {"guide": store.guide}

    @app.put("/api/train/guide")
    def put_train_guide(body: TrainGuideBody) -> dict:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        return save_guide(data_dir, body.guide)

    @app.get("/api/train/vods")
    def list_train_vods() -> list[dict]:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        out: list[dict] = []
        if not data_dir.exists():
            return out
        for child in sorted(data_dir.iterdir()):
            if not child.is_dir():
                continue
            store = load_supervised_store(child)
            if store is None:
                continue
            scan_path = child / "boss_scan.json"
            candidate_count = 0
            if scan_path.exists():
                try:
                    raw = json.loads(scan_path.read_text(encoding="utf-8"))
                    candidate_count = len(raw.get("candidates", []))
                except (json.JSONDecodeError, OSError):
                    candidate_count = 0
            reviewed = sum(
                1
                for r in store.reviews
                if r.verdict in (VERDICT_CORRECT, VERDICT_WRONG)
            )
            out.append(
                {
                    "vod_id": store.vod_id,
                    "title": store.title,
                    "scan_status": store.scan_status,
                    "candidate_count": candidate_count,
                    "reviewed_count": reviewed,
                    "missed_count": len(store.missed_fights),
                }
            )
        return out

    @app.get("/api/train/clips")
    def list_all_manual_clips() -> dict:
        """All saved training clips across VODs (for the trainer list)."""
        from twitch_tiktok_bot.labels.elden_ml.clips import CLIP_KINDS, read_clips

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        clips = read_clips(data_dir)
        clips.sort(key=lambda c: c.get("created_at") or "", reverse=True)
        return {"clips": clips, "kinds": list(CLIP_KINDS), "count": len(clips)}

    @app.delete("/api/train/{vod_id}")
    def remove_train_vod(vod_id: str) -> dict:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        try:
            return delete_train_vod(data_dir, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="VOD not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/train/vods/delete-all")
    def remove_all_train_vods(body: TrainResetBody) -> dict:
        if body.confirm.strip() != "DELETE ALL VODS":
            raise HTTPException(
                status_code=400,
                detail='Type DELETE ALL VODS to remove every upload.',
            )
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        result = delete_all_train_vods(data_dir)
        return result

    @app.post("/api/train/upload")
    async def train_upload(
        file: UploadFile = File(...),
        title: str = Form(""),
    ) -> dict:
        from twitch_tiktok_bot.web.train_jobs import save_upload

        cfg: AppConfig = app.state.config
        data = await file.read()
        if len(data) < 1024:
            raise HTTPException(status_code=400, detail="File too small or empty")
        vod_id, work_dir = save_upload(cfg, file.filename or "upload.mp4", data, title=title)
        return {"vod_id": vod_id, "work_dir": str(work_dir)}

    @app.post("/api/train/{vod_id}/scan")
    def train_scan(vod_id: str) -> dict:
        from twitch_tiktok_bot.web.train_jobs import start_scan

        cfg: AppConfig = app.state.config
        work_dir = cfg.resolve_path(cfg.paths.data_dir) / vod_id
        if not work_dir.exists():
            raise HTTPException(status_code=404, detail="VOD not found")
        status = start_scan(cfg, vod_id)
        return {"vod_id": vod_id, "status": status}

    @app.get("/api/train/{vod_id}/progress")
    def train_vod_progress(vod_id: str) -> dict:
        from twitch_tiktok_bot.web.job_progress import public_progress

        return public_progress(vod_id)

    @app.get("/api/train/jobs/{job_id}")
    def train_job_progress(job_id: str) -> dict:
        from twitch_tiktok_bot.web.job_progress import public_progress

        return public_progress(job_id)

    @app.get("/api/train/{vod_id}")
    def get_train_vod(vod_id: str) -> dict:
        from twitch_tiktok_bot.labels.elden_boss_detect import load_scan_result

        cfg: AppConfig = app.state.config
        try:
            work_dir, video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_supervised_store(work_dir)
        if store is None:
            raise HTTPException(status_code=404, detail="Not a training upload")
        scan = load_scan_result(work_dir)
        return {
            "store": store.to_dict(),
            "scan": scan.to_client_dict() if scan else None,
            "video_url": f"/api/label/{vod_id}/video",
            "video_name": video_path.name,
        }

    @app.post("/api/train/{vod_id}/candidates/{candidate_id}/review")
    def review_candidate(
        vod_id: str, candidate_id: str, body: CandidateReviewBody
    ) -> dict:
        cfg: AppConfig = app.state.config
        work_dir = cfg.resolve_path(cfg.paths.data_dir) / vod_id
        store = load_supervised_store(work_dir)
        if store is None:
            raise HTTPException(status_code=404, detail="Training store not found")
        review = review_for(store, candidate_id)
        review.verdict = body.verdict
        review.wrong_reason = body.wrong_reason.strip()
        review.boss_type = body.boss_type.strip()
        review.notes = body.notes.strip()
        if body.start_sec is not None:
            review.start_sec = round(max(0.0, body.start_sec), 2)
        if body.end_sec is not None:
            review.end_sec = round(max(0.0, body.end_sec), 2)
        review.reviewed_at = datetime.now(timezone.utc).isoformat()
        save_supervised_store(work_dir, store)
        timing_adjusted = review.has_timing_override()
        fb = record_feedback(
            cfg.resolve_path(cfg.paths.data_dir),
            verdict=body.verdict,
            wrong_reason=review.wrong_reason,
            notes=review.notes,
            vod_id=vod_id,
            candidate_id=candidate_id,
            timing_adjusted=timing_adjusted,
        )
        out = review.to_dict()
        out["feedback_changes"] = fb.get("changes", [])
        return out

    @app.post("/api/train/{vod_id}/candidates/{candidate_id}/timing")
    def set_candidate_timing(
        vod_id: str, candidate_id: str, body: TimingBody
    ) -> dict:
        from twitch_tiktok_bot.labels.elden_boss_detect import load_scan_result

        cfg: AppConfig = app.state.config
        work_dir = cfg.resolve_path(cfg.paths.data_dir) / vod_id
        store = load_supervised_store(work_dir)
        if store is None:
            raise HTTPException(status_code=404, detail="Training store not found")
        scan = load_scan_result(work_dir)
        duration = scan.duration_sec if scan else None
        review = review_for(store, candidate_id)

        def _clamp(value: float | None) -> float | None:
            if value is None:
                return None
            value = max(0.0, value)
            if duration:
                value = min(value, duration)
            return round(value, 2)

        review.start_sec = _clamp(body.start_sec)
        review.end_sec = _clamp(body.end_sec)
        if (
            review.start_sec is not None
            and review.end_sec is not None
            and review.end_sec < review.start_sec
        ):
            review.start_sec, review.end_sec = review.end_sec, review.start_sec
        review.reviewed_at = datetime.now(timezone.utc).isoformat()
        save_supervised_store(work_dir, store)
        return review.to_dict()

    @app.post("/api/train/{vod_id}/missed")
    def add_missed_fight(vod_id: str, body: MissedFightBody) -> dict:
        from twitch_tiktok_bot.labels.supervised import MissedFight

        cfg: AppConfig = app.state.config
        work_dir = cfg.resolve_path(cfg.paths.data_dir) / vod_id
        store = load_supervised_store(work_dir)
        if store is None:
            raise HTTPException(status_code=404, detail="Training store not found")
        start, end = body.start_sec, body.end_sec
        if end < start:
            start, end = end, start
        missed = MissedFight(
            id=uuid.uuid4().hex[:10],
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            boss_type=body.boss_type.strip(),
            notes=body.notes.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        store.missed_fights.append(missed)
        save_supervised_store(work_dir, store)
        fb = record_feedback(
            cfg.resolve_path(cfg.paths.data_dir),
            verdict="missed",
            notes=body.notes.strip() or "Boss fight was not detected",
            vod_id=vod_id,
            candidate_id=missed.id,
        )
        out = missed.to_dict()
        out["feedback_changes"] = fb.get("changes", [])
        return out

    @app.get("/api/train/stats")
    def train_stats() -> dict:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        return feedback_stats(data_dir)

    @app.post("/api/train/retune")
    def train_retune() -> dict:
        import threading

        from twitch_tiktok_bot.web.job_progress import finish_job, start_job, update_job

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        job_id = "retune"
        start_job(job_id, kind="retune", message="Starting terminal-first re-scan…")

        def _run() -> None:
            try:
                reports = retune_all_vods(
                    data_dir,
                    clear_reviews=False,
                    on_progress=lambda fields: update_job(job_id, **fields),
                )
                finish_job(
                    job_id,
                    status="done",
                    message=f"Re-scan done ({len(reports)} VOD(s))",
                )
                update_job(job_id, reports=reports)
            except Exception as exc:  # noqa: BLE001
                finish_job(job_id, status="failed", message="Re-scan failed", error=str(exc))

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started", "job_id": job_id}

    @app.get("/api/train/ml/status")
    def train_ml_status() -> dict:
        from twitch_tiktok_bot.labels.elden_ml.config import CLASS_LABELS, load_ml_config
        from twitch_tiktok_bot.labels.elden_ml.dataset import label_counts
        from twitch_tiktok_bot.labels.elden_ml.infer import model_is_ready

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        ml_cfg = load_ml_config(data_dir)
        return {
            "mode": "ml",
            "model_ready": model_is_ready(data_dir),
            "label_counts": label_counts(data_dir),
            "labeled_total": sum(label_counts(data_dir).values()),
            "classes": CLASS_LABELS,
            "config": ml_cfg.to_dict(),
        }

    @app.get("/api/train/ml/frames")
    def train_ml_frames(vod_id: str | None = None, limit: int = 30) -> dict:
        from twitch_tiktok_bot.labels.elden_ml.dataset import list_unlabeled_scan_frames
        from twitch_tiktok_bot.labels.elden_ml.scan import sample_scan_frames

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        # Auto-sample frames for VODs that have video but no scan grid yet.
        targets = []
        if vod_id:
            targets = [data_dir / vod_id]
        else:
            targets = [
                c
                for c in sorted(data_dir.iterdir())
                if c.is_dir() and c.name != "reference" and load_supervised_store(c)
            ]
        ffmpeg = cfg.render.ffmpeg_path or "ffmpeg"
        for work_dir in targets:
            frames = work_dir / "boss_scan_frames"
            if frames.is_dir() and any(frames.glob("frame_*.jpg")):
                continue
            video = None
            for pat in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
                found = list(work_dir.glob(pat))
                if found:
                    video = found[0]
                    break
            if video is None:
                continue
            try:
                sample_scan_frames(video, work_dir, interval_sec=2.0, ffmpeg=ffmpeg)
            except Exception:  # noqa: BLE001
                continue
        items = list_unlabeled_scan_frames(data_dir, vod_id=vod_id, limit=limit)
        return {"frames": items, "count": len(items)}

    @app.get("/api/train/ml/frame-image")
    def train_ml_frame_image(path: str) -> FileResponse:
        """Serve a scan frame for the labeler (must live under data/)."""
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir).resolve()
        target = Path(path).resolve()
        try:
            target.relative_to(data_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid path") from exc
        if not target.exists() or target.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise HTTPException(status_code=404, detail="Frame not found")
        return FileResponse(target)

    @app.post("/api/train/ml/label")
    def train_ml_label(body: MlLabelBody) -> dict:
        from twitch_tiktok_bot.labels.elden_ml.config import CLASS_TO_IDX
        from twitch_tiktok_bot.labels.elden_ml.dataset import append_label, label_counts

        if body.label not in CLASS_TO_IDX:
            raise HTTPException(status_code=400, detail=f"Invalid label {body.label}")
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir).resolve()
        source = Path(body.path).resolve()
        try:
            source.relative_to(data_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid path") from exc
        try:
            row = append_label(
                data_dir,
                label=body.label,
                source_path=source,
                vod_id=body.vod_id,
                time_sec=body.time_sec,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "row": row, "label_counts": label_counts(data_dir)}

    @app.post("/api/train/ml/train")
    def train_ml_train(body: MlTrainBody) -> dict:
        import threading

        from twitch_tiktok_bot.labels.elden_ml.infer import clear_model_cache
        from twitch_tiktok_bot.labels.elden_ml.train import train_model
        from twitch_tiktok_bot.web.job_progress import finish_job, start_job, update_job

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        job_id = "ml_train"
        start_job(job_id, kind="ml_train", message="Starting model training…")

        def _run() -> None:
            try:
                result = train_model(
                    data_dir,
                    epochs=body.epochs,
                    min_labels=body.min_labels,
                    on_progress=lambda fields: update_job(job_id, **fields),
                )
                clear_model_cache()
                finish_job(
                    job_id,
                    status="done",
                    message=f"Trained — val acc {result.get('accuracy', 0):.0%}",
                )
                update_job(job_id, result=result, pct=100.0)
            except Exception as exc:  # noqa: BLE001
                finish_job(job_id, status="failed", message="Training failed", error=str(exc))

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started", "job_id": job_id}

    @app.post("/api/train/ml/pack/merica")
    def train_ml_pack_merica(body: MericaPackBody) -> dict:
        """Import Merica hard-negatives + dual Tree Sentinel fight labels."""
        from twitch_tiktok_bot.labels.elden_ml.hardneg_packs import (
            apply_merica_hardneg_pack,
        )

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        ffmpeg = cfg.render.ffmpeg_path or "ffmpeg"
        try:
            return apply_merica_hardneg_pack(
                data_dir,
                force=body.force,
                extract_video=body.extract_video,
                ffmpeg=ffmpeg,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/train/{vod_id}/clips")
    def list_manual_clips(vod_id: str) -> dict:
        from twitch_tiktok_bot.labels.elden_ml.clips import CLIP_KINDS, read_clips

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        work_dir = data_dir / vod_id
        if not work_dir.exists():
            raise HTTPException(status_code=404, detail="VOD not found")
        clips = read_clips(data_dir, vod_id=vod_id)
        return {"vod_id": vod_id, "clips": clips, "kinds": list(CLIP_KINDS)}

    @app.post("/api/train/{vod_id}/clips")
    def create_manual_clip(vod_id: str, body: ManualClipBody) -> dict:
        from twitch_tiktok_bot.labels.elden_ml.clips import save_manual_clip

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        if not (data_dir / vod_id).exists():
            raise HTTPException(status_code=404, detail="VOD not found")
        ffmpeg = cfg.render.ffmpeg_path or "ffmpeg"
        try:
            return save_manual_clip(
                data_dir,
                vod_id=vod_id,
                start_sec=body.start_sec,
                end_sec=body.end_sec,
                kind=body.kind,
                notes=body.notes,
                extract_video=body.extract_video,
                expand_frames=body.expand_frames,
                ffmpeg=ffmpeg,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/train/{vod_id}/clips/{clip_id}")
    def edit_manual_clip(vod_id: str, clip_id: str, body: ManualClipUpdateBody) -> dict:
        from twitch_tiktok_bot.labels.elden_ml.clips import update_manual_clip

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        if not (data_dir / vod_id).exists():
            raise HTTPException(status_code=404, detail="VOD not found")
        ffmpeg = cfg.render.ffmpeg_path or "ffmpeg"
        try:
            result = update_manual_clip(
                data_dir,
                clip_id,
                start_sec=body.start_sec,
                end_sec=body.end_sec,
                kind=body.kind,
                notes=body.notes,
                extract_video=body.extract_video,
                expand_frames=body.expand_frames,
                ffmpeg=ffmpeg,
            )
            if result["clip"].get("vod_id") != vod_id:
                raise HTTPException(status_code=404, detail="Clip not on this VOD")
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Clip not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/train/{vod_id}/clips/{clip_id}")
    def remove_manual_clip(vod_id: str, clip_id: str) -> dict:
        from twitch_tiktok_bot.labels.elden_ml.clips import delete_manual_clip

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        try:
            return delete_manual_clip(data_dir, clip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Clip not found") from exc

    @app.get("/api/train/{vod_id}/clips/{clip_id}/video")
    def manual_clip_video(vod_id: str, clip_id: str) -> FileResponse:
        from twitch_tiktok_bot.labels.elden_ml.clips import read_clips

        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        clip = next((c for c in read_clips(data_dir, vod_id=vod_id) if c.get("id") == clip_id), None)
        if not clip or not clip.get("clip_path"):
            raise HTTPException(status_code=404, detail="Clip video not found")
        path = Path(clip["clip_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Clip file missing")
        return FileResponse(path, media_type="video/mp4", filename=f"{clip_id}.mp4")

    @app.post("/api/train/reset")
    def train_reset(body: TrainResetBody) -> dict:
        if body.confirm.strip() != "START FRESH":
            raise HTTPException(
                status_code=400,
                detail='Type START FRESH in the confirmation box to erase training.',
            )
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        return reset_elden_training(data_dir, clear_scan_cache=True)

    return app


def run_server(config: AppConfig | None = None) -> None:
    import uvicorn

    cfg = config or load_config()
    app = create_app(cfg)
    base = f"http://{cfg.web.host}:{cfg.web.port}"
    print(f"  Preview UI     {base}/")
    print(f"  Fight labeler  {base}/label")
    print(f"  Elden trainer  {base}/train")
    print(f"  Apex trainer   {base}/literacy")
    print(f"  Auth / YouTube {base}/auth")
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="info")


if __name__ == "__main__":
    run_server()
