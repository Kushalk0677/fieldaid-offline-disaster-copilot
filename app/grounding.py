from __future__ import annotations

from pathlib import Path

from app.image_classifier import classify_disaster_image
from app.models import GuidanceCitation, UploadedImageInfo
from app.ollama_client import call_gemma4_grounding_image
from app.rag import retrieve_guidance
from app.tools import safety_flags


async def analyze_grounding(
    question: str,
    scenario_type: str,
    image_info: UploadedImageInfo | None,
    image_path: Path | None,
    model: str,
) -> dict:
    citations = retrieve_guidance(question + " " + (image_info.filename if image_info else ""), scenario_type)  # type: ignore[arg-type]
    if image_info and image_path:
        classifier = classify_disaster_image(image_path)
        grounded_question = _question_with_classifier_context(question, classifier)
        try:
            raw = await call_gemma4_grounding_image(grounded_question, image_path, model)
            result = _coerce_grounding_result(raw, citations, image_info, "gemma4-ollama", classifier)
        except Exception as exc:
            result = _fallback_grounding_result(question, scenario_type, citations, image_info, image_path, exc, classifier)
    else:
        result = _text_grounding_result(question, scenario_type, citations)
    return result


def _text_grounding_result(question: str, scenario_type: str, citations: list[GuidanceCitation]) -> dict:
    lowered = question.lower()
    unsafe = any(term in lowered for term in ["bridge", "cross", "civilians", "unsafe", "floodwater", "collapse", "road", "fire", "smoke"])
    if unsafe:
        answer = "Do not treat this as safe without authorized human verification. FieldAid recommends isolating the hazard, confirming conditions, and escalating to incident command or qualified specialists."
        status = "high"
    else:
        answer = "FieldAid can summarize relevant guidance, but field conditions and responsible authorities must verify the decision before action."
        status = "unknown"
    return {
        "status": status,
        "scene_type": "unknown",
        "answer": answer,
        "visual_evidence": [],
        "recommended_actions": ["Verify field conditions before action."],
        "discord_message": build_discord_message(status, "unknown", answer),
        "confidence": 0.45,
        "verification_flags": safety_flags(question, scenario_type),  # type: ignore[arg-type]
        "citations": [citation.model_dump(mode="json") for citation in citations],
        "image": None,
        "tool_trace": ["retrieve_guidance", "check_unsupported_claims", "apply_safety_refusal", "draft_discord_message"],
        "engine": "text-grounding",
    }


def _fallback_grounding_result(
    question: str,
    scenario_type: str,
    citations: list[GuidanceCitation],
    image_info: UploadedImageInfo,
    image_path: Path | None = None,
    error: Exception | None = None,
    classifier: dict | None = None,
) -> dict:
    lowered = f"{question} {image_info.filename}".lower()
    classifier = classifier or classify_disaster_image(image_path) if image_path else _classifier_unavailable()
    local = _infer_scene_from_classifier(classifier)
    if local["scene_type"] == "unknown":
        local = _infer_scene_from_image(image_path)
    scene_type = local["scene_type"]
    if scene_type == "unknown" and any(term in lowered for term in ["fire", "flame", "smoke", "burning"]):
        scene_type = "fire"
        local = {
            "scene_type": scene_type,
            "status": "high",
            "confidence": 0.46,
            "visual_evidence": ["Filename or question references fire, flame, smoke, or burning."],
            "recommended_actions": ["Keep civilians away from smoke or flame.", "Escalate to fire response or incident command."],
        }
    status = local["status"]
    reason = _friendly_gemma_error(error)
    if scene_type == "fire":
        answer = (
            f"Gemma image analysis is unavailable ({reason}), so FieldAid used a local visual fallback. "
            "The photo or filename suggests a possible fire or smoke hazard; isolate the area and request human verification."
        )
    elif scene_type in {"building_damage", "road_damage", "flood"}:
        readable_scene = scene_type.replace("_", " ")
        answer = (
            f"Gemma image analysis is unavailable ({reason}), so FieldAid used the local MEDIC disaster classifier. "
            f"The photo is {classifier.get('confidence_band', 'uncertain')} for {readable_scene}; treat this as an unverified hazard and escalate for human review."
        )
    elif scene_type == "smoke":
        answer = (
            f"Gemma image analysis is unavailable ({reason}), so FieldAid used a local visual fallback. "
            "The photo contains haze/smoke-like regions; treat air quality and fire-source status as unverified."
        )
    else:
        answer = (
            f"Gemma image analysis is unavailable ({reason}). FieldAid saved the photo but could not identify a clear disaster type locally; manual review is required."
        )
    return {
        "status": status,
        "scene_type": scene_type,
        "answer": answer,
        "visual_evidence": [
            f"Image uploaded: {image_info.filename} ({image_info.content_type}, {image_info.size_bytes} bytes).",
            *local["visual_evidence"],
        ],
        "recommended_actions": local["recommended_actions"],
        "discord_message": build_discord_message(status, scene_type, answer),
        "confidence": local["confidence"],
        "verification_flags": safety_flags(f"{question} {scene_type}", scenario_type),  # type: ignore[arg-type]
        "citations": [citation.model_dump(mode="json") for citation in citations],
        "image": image_info.model_dump(mode="json"),
        "classifier": classifier,
        "tool_trace": ["save_photo", "run_medic_classifier", "retrieve_guidance", "gemma_image_unavailable", "local_visual_fallback", "apply_safety_refusal", "draft_discord_message"],
        "engine": "gemma-unavailable/local-classifier-fallback",
    }


def _infer_scene_from_classifier(classifier: dict) -> dict:
    if not classifier.get("available"):
        return _base_local_scene([f"Local MEDIC classifier unavailable: {classifier.get('reason', 'unknown reason')}"])
    label = str(classifier.get("top_label") or "unknown")
    confidence = float(classifier.get("top_confidence") or 0.0)
    band = str(classifier.get("confidence_band") or "uncertain")
    if label == "unknown" or confidence < 0.45:
        return _base_local_scene([f"Local MEDIC classifier was uncertain; top score was {label} at {confidence:.0%}."])

    status_by_label = {
        "building_damage": "critical" if confidence >= 0.70 else "high",
        "fire": "critical" if confidence >= 0.70 else "high",
        "flood": "high",
        "road_damage": "high",
    }
    actions_by_label = {
        "building_damage": ["Keep people clear of the structure.", "Request structural or urban search-and-rescue verification.", "Check for trapped or injured people from a safe distance."],
        "fire": ["Keep civilians away from flame or smoke.", "Escalate to fire response or incident command.", "Check for evacuation and medical standby needs."],
        "flood": ["Keep civilians away from floodwater.", "Check alternate access routes.", "Escalate water rescue or road closure decisions to incident command."],
        "road_damage": ["Do not declare the route safe from imagery.", "Mark the route for responder verification.", "Identify alternate access for emergency vehicles."],
    }
    return {
        "scene_type": label,
        "status": status_by_label.get(label, "high"),
        "confidence": confidence,
        "visual_evidence": [f"Local MEDIC classifier: {band} {label.replace('_', ' ')} ({confidence:.0%})."],
        "recommended_actions": actions_by_label.get(label, ["Keep civilians away from uncertain hazards.", "Escalate to the responsible authority for verification."]),
    }


def _infer_scene_from_image(image_path: Path | None) -> dict:
    base = _base_local_scene(["No local visual signal was strong enough to classify the photo."])
    if image_path is None or not image_path.exists():
        return base
    try:
        import cv2
        import numpy as np
    except ImportError:
        base["visual_evidence"] = ["OpenCV local visual fallback is unavailable in this environment."]
        return base

    image = cv2.imread(str(image_path))
    if image is None:
        base["visual_evidence"] = ["Uploaded file could not be decoded by the local visual fallback."]
        return base

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 80, 120])
    upper_orange = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    fire_ratio = float(cv2.countNonZero(fire_mask)) / float(image.shape[0] * image.shape[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    smoke_mask = ((gray > 70) & (gray < 220) & (saturation < 70)).astype("uint8")
    smoke_ratio = float(smoke_mask.sum()) / float(image.shape[0] * image.shape[1])

    if fire_ratio > 0.025:
        return {
            "scene_type": "fire",
            "status": "high",
            "confidence": min(0.82, 0.45 + fire_ratio * 8),
            "visual_evidence": [f"Local fallback found flame-color regions in {fire_ratio:.1%} of the image."],
            "recommended_actions": ["Keep civilians away from flame or smoke.", "Escalate to fire response or incident command.", "Check for evacuation and medical standby needs."],
        }
    if smoke_ratio > 0.18:
        return {
            "scene_type": "smoke",
            "status": "high",
            "confidence": min(0.72, 0.35 + smoke_ratio),
            "visual_evidence": [f"Local fallback found smoke/haze-like regions in {smoke_ratio:.1%} of the image."],
            "recommended_actions": ["Treat air quality as uncertain.", "Check for a fire source.", "Keep civilians clear until verified."],
        }
    return base


def _base_local_scene(visual_evidence: list[str]) -> dict:
    return {
        "scene_type": "unknown",
        "status": "unknown",
        "confidence": 0.35,
        "visual_evidence": visual_evidence,
        "recommended_actions": ["Keep civilians away from uncertain hazards.", "Escalate to the responsible authority for verification."],
    }


def _question_with_classifier_context(question: str, classifier: dict) -> str:
    if not classifier.get("available"):
        return question
    predictions = ", ".join(
        f"{item['label']} {float(item['confidence']):.0%}"
        for item in classifier.get("predictions", [])[:4]
    )
    return (
        f"{question}\n\n"
        "Local MEDIC classifier evidence from the uploaded image: "
        f"{predictions}. Use this only as supporting evidence; do not declare any route, bridge, or structure safe."
    )


def _classifier_unavailable() -> dict:
    return {
        "available": False,
        "model": "fieldaid-medic-image-classifier",
        "checkpoint": None,
        "classes": [],
        "predictions": [],
        "top_label": None,
        "top_confidence": 0.0,
        "confidence_band": "unavailable",
        "reason": "no image path supplied",
    }


def _friendly_gemma_error(error: Exception | None) -> str:
    if error is None:
        return "model call failed"
    text = str(error).lower()
    if "unable to connect" in text or "connection" in text or "connect" in text:
        return "Ollama is not reachable"
    if "404" in text or "not found" in text:
        return "selected model is not available in Ollama"
    return "model call failed"


def _coerce_grounding_result(
    raw: dict,
    citations: list[GuidanceCitation],
    image_info: UploadedImageInfo,
    engine: str,
    classifier: dict | None = None,
) -> dict:
    status = str(raw.get("status", "unknown")).lower()
    if status not in {"critical", "high", "medium", "low", "unknown"}:
        status = "unknown"
    scene_type = str(raw.get("scene_type", "unknown")).lower() or "unknown"
    answer = str(raw.get("answer") or "Image reviewed; field verification is required.")
    discord_message = str(raw.get("discord_message") or build_discord_message(status, scene_type, answer))
    return {
        "status": status,
        "scene_type": scene_type,
        "answer": answer,
        "visual_evidence": [str(item) for item in raw.get("visual_evidence", [])],
        "recommended_actions": [str(item) for item in raw.get("recommended_actions", [])],
        "discord_message": discord_message,
        "confidence": max(0.0, min(float(raw.get("confidence", 0.6)), 1.0)),
        "verification_flags": [str(item) for item in raw.get("verification_flags", [])]
        or ["Photo-grounded status requires human verification."],
        "citations": [citation.model_dump(mode="json") for citation in citations],
        "image": image_info.model_dump(mode="json"),
        "classifier": classifier or _classifier_unavailable(),
        "tool_trace": ["save_photo", "run_medic_classifier", "analyze_photo_with_gemma", "retrieve_guidance", "apply_safety_check", "draft_discord_message"],
        "engine": engine,
    }


def build_discord_message(status: str, scene_type: str, answer: str) -> str:
    return f"**FieldAid status: {status.upper()}** | Scene: {scene_type}. {answer} Human verification required before action."
