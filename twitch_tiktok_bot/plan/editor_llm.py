"""LLM-based edit planning with JSON output."""

from __future__ import annotations

import json
import os
from typing import Any

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan
from twitch_tiktok_bot.plan.rules import create_rule_based_plan

SYSTEM_PROMPT = """You are a short-form video editor for TikTok.
Given analysis of a Twitch clip, output a JSON edit plan only.

Goals:
- One coherent 15-60 second moment
- Prefer segments with loud peaks and exciting speech
- Add light entertainment: 1-3 zoom effects on reactions, bold hook text
- Remove long dead air conceptually by picking a tight window

Output JSON schema:
{
  "target_duration_sec": number,
  "segments": [{"start": number, "end": number, "reason": string}],
  "effects": [{"t": number, "type": "zoom"|"caption_emphasis"|"sfx", "scale": number, "duration": number, "text": string, "asset": string, "volume": number}],
  "hook_text": string,
  "hashtags": [string],
  "caption_style": "bold"|"karaoke"
}
"""


def _compact_analysis(analysis: ClipAnalysis) -> dict[str, Any]:
    data = analysis.to_dict()
    # Trim transcript for token limits
    data["transcript_segments"] = data["transcript_segments"][:40]
    data["loud_peaks"] = data["loud_peaks"][:20]
    return data


def create_llm_plan(analysis: ClipAnalysis, config: AppConfig) -> EditPlan | None:
    llm = config.llm
    api_key = os.getenv(llm.api_key_env, "")
    if not llm.enabled or not api_key:
        return None

    from openai import OpenAI

    client_kwargs: dict = {"api_key": api_key}
    if llm.base_url:
        client_kwargs["base_url"] = llm.base_url
    client = OpenAI(**client_kwargs)

    user_content = json.dumps(_compact_analysis(analysis), indent=2)
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
    plan = EditPlan.from_dict(data)

    # Basic validation
    if not plan.segments:
        return None
    for seg in plan.segments:
        if seg.end <= seg.start or seg.end > analysis.duration + 0.5:
            return None
    return plan


def create_edit_plan(analysis: ClipAnalysis, config: AppConfig) -> EditPlan:
    plan = create_llm_plan(analysis, config)
    if plan is not None:
        print("  [plan] using LLM edit plan")
        return plan
    print("  [plan] using rule-based edit plan")
    return create_rule_based_plan(analysis, config)
