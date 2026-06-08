"""LLM-based edit planning (Cursor SDK or OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan
from twitch_tiktok_bot.plan.action import summarize_clip_action
from twitch_tiktok_bot.plan.game_profiles import apex_fight_windows, detect_game_profile
from twitch_tiktok_bot.plan.moments import rank_moments
from twitch_tiktok_bot.plan.refine import build_catchy_plan, refine_plan
from twitch_tiktok_bot.plan.rules import create_rule_based_plan

SYSTEM_PROMPT = """You are a viral TikTok gaming clip editor. Output ONLY valid JSON.

Your job: turn Twitch VOD energy into a scroll-stopping 18-35 second short.

Strategy:
1. HOOK (segment 1): Open with the single highest-energy payoff — a reaction, clutch, or punchline. Never open with setup/menu talk ("how do I", "let me", "testing").
2. BODY: 2-4 micro-clips (2-7 sec each) that build context then land the payoff again.
3. Cut all dead air — use ranked_moments and silence_ranges to avoid boring windows.
4. hook_text: curiosity-gap title ("He actually hit it", "Ranked was a mistake") — NEVER copy the first transcript line verbatim.
5. Zoom only on the top 1-2 reaction peaks (scale 1.15-1.2).

Output JSON schema:
{
  "target_duration_sec": number,
  "segments": [{"start": number, "end": number, "reason": string}],
  "effects": [{"t": number, "type": "zoom"|"caption_emphasis"|"sfx", "scale": number, "duration": number, "text": string, "asset": string, "volume": number}],
  "hook_text": string,
  "hashtags": [string],
  "caption_style": "bold"|"karaoke"
}

Hard rules:
- segments: 2-7 seconds each, non-overlapping, within clip duration
- first segment = highest score moment from ranked_moments
- total runtime 18-35 sec unless clip is shorter
- NEVER use segments that are mostly setup/menu talk (how do I, edit, let me do 100, requeue)
- If action_assessment.has_combat is false: use ONLY 1 short segment from best_action_window, do NOT pad with talk

Apex Legends (when game_profile is apex):
- REQUIRE knocks/cracks/gunfire audio bursts — "nice" or "here we go" alone is NOT a highlight
- Prioritize apex_fight_windows with real rapid peak clusters
- Skip lobby/menu/end-screen/requeue/firing-range testing footage
- Hook examples: "1v3 and he does THIS", "Squad wipe incoming", "Cracked — then this"
"""


def _compact_analysis(analysis: ClipAnalysis, config: AppConfig) -> dict[str, Any]:
    data = analysis.to_dict()
    data["transcript_segments"] = data["transcript_segments"][:40]
    data["loud_peaks"] = data["loud_peaks"][:20]
    data["silence_ranges"] = data["silence_ranges"][:15]
    profile = detect_game_profile(analysis, config.editing.game_profile)
    data["game_profile"] = profile
    if profile == "apex":
        data["apex_fight_windows"] = [
            {"start": s, "end": e, "score": round(sc, 1)}
            for s, e, sc in apex_fight_windows(analysis)[:6]
        ]
    moments = rank_moments(analysis, limit=8, game_profile=config.editing.game_profile)
    data["ranked_moments"] = [
        {
            "start": m.start,
            "end": m.end,
            "score": round(m.score, 2),
            "reason": m.reason,
            "quote": m.quote[:80],
        }
        for m in moments
    ]
    if moments:
        data["best_payoff_quote"] = moments[0].quote
    action = summarize_clip_action(analysis, config.editing.game_profile)
    data["action_assessment"] = {
        "has_combat": action.has_combat,
        "combat_score": round(action.combat_score, 2),
        "setup_ratio": round(action.setup_ratio, 2),
        "warning": action.warning,
        "best_action_window": (
            {"start": action.best_window[0], "end": action.best_window[1]}
            if action.best_window
            else None
        ),
    }
    return data


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return json.loads(fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Response did not contain JSON: {text[:500]}...")


def _validate_plan(plan: EditPlan, analysis: ClipAnalysis) -> EditPlan | None:
    if not plan.segments:
        return None
    for seg in plan.segments:
        if seg.end <= seg.start or seg.end > analysis.duration + 0.5:
            return None
    return plan


def create_cursor_plan(analysis: ClipAnalysis, config: AppConfig) -> EditPlan | None:
    llm = config.llm
    api_key = os.getenv(llm.cursor_api_key_env, "")
    if not llm.enabled or not api_key:
        return None

    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    from cursor_sdk.errors import AuthenticationError

    from twitch_tiktok_bot.plan.cursor_bridge import launch_sdk_client

    user_content = json.dumps(_compact_analysis(analysis, config), indent=2)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Create a TikTok edit plan for this Twitch clip analysis:\n"
        f"{user_content}\n\n"
        "Respond with ONLY the JSON object."
    )

    try:
        with launch_sdk_client(
            workspace=str(config.project_root),
            cursor_api_key=api_key,
        ) as client:
            result = Agent.prompt(
                prompt,
                AgentOptions(
                    api_key=api_key,
                    model=llm.cursor_model,
                    local=LocalAgentOptions(
                        cwd=str(config.project_root),
                        setting_sources=[],
                    ),
                ),
                client=client,
            )
    except AuthenticationError as exc:
        raise RuntimeError(
            "Cursor API key invalid. Get one at https://cursor.com/dashboard/integrations "
            f"and set {llm.cursor_api_key_env} in .env"
        ) from exc

    if result.status == "error":
        print(f"  [plan] Cursor agent error: {result.result}")
        return None

    raw = result.result or "{}"
    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  [plan] Cursor returned invalid JSON: {exc}")
        return None

    plan = _validate_plan(EditPlan.from_dict(data), analysis)
    return refine_plan(plan, analysis, config) if plan else None


def create_openai_plan(analysis: ClipAnalysis, config: AppConfig) -> EditPlan | None:
    llm = config.llm
    api_key = os.getenv(llm.api_key_env, "")
    if not llm.enabled or not api_key:
        return None

    from openai import OpenAI

    client_kwargs: dict = {"api_key": api_key}
    if llm.base_url:
        client_kwargs["base_url"] = llm.base_url
    client = OpenAI(**client_kwargs)

    user_content = json.dumps(_compact_analysis(analysis, config), indent=2)
    response = client.chat.completions.create(
        model=llm.model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Create a TikTok edit plan for this Twitch clip analysis:\n"
                    f"{user_content}"
                ),
            },
        ],
        temperature=0.4,
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    plan = _validate_plan(EditPlan.from_dict(data), analysis)
    return refine_plan(plan, analysis, config) if plan else None


def create_llm_plan(analysis: ClipAnalysis, config: AppConfig) -> EditPlan | None:
    provider = config.llm.provider.lower()
    if provider == "cursor":
        return create_cursor_plan(analysis, config)
    return create_openai_plan(analysis, config)


def create_edit_plan(
    analysis: ClipAnalysis,
    config: AppConfig,
    work_dir: Path | None = None,
) -> EditPlan:
    from twitch_tiktok_bot.labels.curator import (
        load_curator_references_for_work_dir,
        sync_curator_reference_yaml,
    )
    from twitch_tiktok_bot.plan.action_cuts import apply_action_cuts_to_plan
    from twitch_tiktok_bot.labels.fights import load_fight_labels, plan_from_fight_labels
    from twitch_tiktok_bot.progress import step

    if work_dir is not None:
        store = load_fight_labels(work_dir)
        if store and store.fights:
            labeled = plan_from_fight_labels(store, analysis, config)
            if labeled is not None:
                print(
                    f"  [plan] using {len(labeled.segments)} curator-labeled "
                    f"fight window(s) from fight_labels.json"
                )
                if config.editing.action_cut_enabled:
                    labeled = apply_action_cuts_to_plan(
                        labeled, analysis, config, work_dir=work_dir
                    )
                    if labeled.segments:
                        print(
                            f"  [plan] action-cut edit: {len(labeled.segments)} "
                            f"beats, {labeled.target_duration_sec:.0f}s total"
                        )
                return labeled

        curator_refs = load_curator_references_for_work_dir(work_dir)
        if curator_refs:
            print(
                f"  [plan] {len(curator_refs)} fight reference(s) from "
                f"{work_dir.name} — anchoring clips to your start/end timing"
            )
            data_dir = config.resolve_path(config.paths.data_dir)
            sync_curator_reference_yaml(
                data_dir,
                config.resolve_path("data/reference/apex_curator_fights.json"),
            )

    if config.editing.scan_full_pov:
        with step("  [plan] scanning full POV for montage clips", heartbeat_sec=0):
            return create_rule_based_plan(analysis, config, work_dir=work_dir)

    plan: EditPlan | None = None
    if config.llm.enabled:
        label = "Cursor" if config.llm.provider.lower() == "cursor" else "LLM"
        try:
            with step(f"  [plan] asking {label} for edit plan", heartbeat_sec=20):
                plan = create_llm_plan(analysis, config)
        except Exception as exc:
            print(f"  [plan] LLM failed ({exc}), falling back to rules")
            plan = None

    if plan is not None:
        print(f"  [plan] using {label} edit plan")
        return plan

    with step("  [plan] building rule-based edit plan", heartbeat_sec=0):
        plan = create_rule_based_plan(analysis, config, work_dir=work_dir)
        return refine_plan(plan, analysis, config)
