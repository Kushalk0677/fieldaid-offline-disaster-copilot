from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from app.models import ScenarioType


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


SYSTEM_PROMPT = """You are FieldAid, an offline disaster-response copilot.
You help trained responders summarize field reports, prioritize resources, and stay grounded in provided emergency guidance.
Return only valid JSON. Do not include markdown.
Never claim to replace emergency services, clinicians, engineers, or incident command.
Flag medical, structural, and life-safety recommendations for human verification."""


def build_prompt(
    scenario_type: ScenarioType,
    note_text: str,
    location: str,
    guidance_context: str,
) -> str:
    return f"""
Scenario type: {scenario_type}
Location: {location}

Responder note:
{note_text or "[No text note provided. Use the image if present and be explicit about uncertainty.]"}

Offline guidance excerpts:
{guidance_context}

Return JSON with this shape:
{{
  "summary": "one sentence",
  "urgency": "critical|high|medium|low",
  "extracted_facts": ["facts directly visible in note/image"],
  "inferred_recommendations": ["recommendations inferred from facts and guidance"],
  "action_plan": [
    {{"priority":"critical|high|medium|low","title":"short title","rationale":"why","owner":"role","next_step":"specific action"}}
  ],
  "supply_request": [
    {{"item":"resource","quantity":"estimate or count","reason":"why needed"}}
  ],
  "sms_update": "short radio/SMS-safe operational update",
  "confidence": 0.0,
  "verification_flags": ["human verification requirements"]
}}
""".strip()


async def call_gemma4(
    scenario_type: ScenarioType,
    note_text: str,
    location: str,
    guidance_context: str,
    model: str,
    image_path: Path | None = None,
    before_image_path: Path | None = None,
    timeout_seconds: float = 90,
) -> dict:
    user_message: dict = {
        "role": "user",
        "content": build_prompt(scenario_type, note_text, location, guidance_context),
    }
    images = []
    if before_image_path and before_image_path.exists():
        images.append(base64.b64encode(before_image_path.read_bytes()).decode("ascii"))
    if image_path and image_path.exists():
        images.append(base64.b64encode(image_path.read_bytes()).decode("ascii"))
    if images:
        user_message["images"] = images
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        "options": {"temperature": 0.1},
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return parse_json_response(content)


async def call_gemma4_video_frames(
    frame_paths: list[Path],
    location: str,
    model: str,
    timeout_seconds: float = 120,
) -> dict:
    prompt = f"""
You are FieldAid's video disaster analyst. You will receive sampled frames from one uploaded field video.
Identify the likely disaster or hazard type from the frames.

Location: {location}

Return only valid JSON with this shape:
{{
  "disaster_type": "fire|flood|road_damage|building_damage|crowd_risk|smoke|unknown",
  "summary": "one sentence describing what the frames appear to show",
  "confidence": 0.0,
  "visual_evidence": ["specific visible evidence from the frames"],
  "recommended_focus": ["response focus areas such as evacuation, fire suppression, route closure, medical standby"],
  "verification_flags": ["human verification requirements"]
}}

Do not say a route, bridge, building, or fire scene is safe. If uncertain, say unknown and require human review.
""".strip()
    images = [base64.b64encode(path.read_bytes()).decode("ascii") for path in frame_paths[:6] if path.exists()]
    if not images:
        raise ValueError("No sampled frames available for Gemma video analysis.")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": images},
        ],
        "options": {"temperature": 0.1},
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return parse_json_response(content)


async def call_gemma4_grounding_image(
    question: str,
    image_path: Path,
    model: str,
    timeout_seconds: float = 90,
) -> dict:
    prompt = f"""
You are FieldAid's photo-grounded safety analyst.
Inspect the uploaded image and answer the responder's question cautiously.

Responder question:
{question}

Return only valid JSON with this shape:
{{
  "status": "critical|high|medium|low|unknown",
  "scene_type": "fire|flood|road_damage|building_damage|medical|shelter|unknown",
  "answer": "short status answer grounded in visible evidence",
  "visual_evidence": ["specific things visible in the image"],
  "recommended_actions": ["safe operational next steps"],
  "discord_message": "short Discord-ready update for responders",
  "confidence": 0.0,
  "verification_flags": ["human verification requirements"]
}}

Do not declare a road, bridge, building, fire, or medical situation safe. If uncertain, say unknown and require human review.
""".strip()
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
            },
        ],
        "options": {"temperature": 0.1},
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return parse_json_response(content)


def parse_json_response(content: str) -> dict:
    content = content.strip()
    if not content:
        raise ValueError("Empty model response")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    return parsed
