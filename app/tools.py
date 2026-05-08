from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from app.models import ActionItem, AnalysisResult, GuidanceCitation, ScenarioType, SupplyItem, TimelineEvent, UploadedImageInfo, Urgency


HIGH_RISK_TERMS = {
    "medical": ["insulin", "diabetic", "injury", "injured", "medicine", "patient", "elderly"],
    "structural": ["bridge", "collapse", "crack", "road blocked", "building", "power line"],
    "life_safety": ["trapped", "evacuate", "evacuation", "flood", "fire", "landslide"],
}


def create_incident_report(fields: dict) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario_type": fields.get("scenario_type", "combined"),
        "location": fields.get("location", "Unknown location"),
        "urgency": fields.get("urgency", "medium"),
        "summary": fields.get("summary", "Incident requires field review."),
        "status": fields.get("status", "new"),
    }


def create_supply_request(
    items: list[SupplyItem],
    urgency: Urgency,
    location: str,
) -> dict:
    return {
        "location": location,
        "urgency": urgency,
        "items": [item.model_dump() for item in items],
        "delivery_note": "Confirm route access and handoff contact before dispatch.",
    }


def generate_sms_update(
    summary: str,
    urgency: Urgency,
    location: str,
    action_plan: list[ActionItem],
    audience: str,
    role: str = "district operations",
    max_len: int = 320,
) -> str:
    first_action = action_plan[0].next_step if action_plan else "Field verification requested."
    role_label = role.replace("_", " ")
    sms = f"{urgency.upper()} FieldAid update for {audience} ({role_label}): {location}. {summary} Next: {first_action}"
    return sms[: max_len - 1].rstrip() + "." if len(sms) > max_len else sms


def update_incident_status(report: dict, status: str) -> dict:
    updated = dict(report)
    updated["status"] = status
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated


def safety_flags(text: str, scenario_type: ScenarioType) -> list[str]:
    lowered = text.lower()
    flags: list[str] = []
    for category, terms in HIGH_RISK_TERMS.items():
        if any(term in lowered for term in terms):
            flags.append(f"{category.replace('_', ' ').title()} guidance requires human verification.")
    if scenario_type == "damage":
        flags.append("Structural damage assessment requires inspection by authorized personnel.")
    if not flags:
        flags.append("Verify field conditions before dispatching resources.")
    return sorted(set(flags))


def fallback_analysis(
    scenario_type: ScenarioType,
    note_text: str,
    location: str,
    audience: str,
    citations: list[GuidanceCitation],
    model: str,
    image: UploadedImageInfo | None = None,
    before_image: UploadedImageInfo | None = None,
    role: str = "district operations",
    sms_language: str = "English",
) -> AnalysisResult:
    text = _combine_note_and_image_context(note_text, image, before_image)
    lowered = text.lower()
    people_match = re.search(r"(\d+)\s*(people|persons|residents|evacuees)", lowered)
    water_match = re.search(r"water(?:\s+left)?(?:\s+for)?\s*(\d+)\s*(hour|hr|hrs|hours)", lowered)
    facts: list[str] = []
    if people_match:
        facts.append(f"Reported population: {people_match.group(1)} people.")
    if "elderly" in lowered:
        facts.append("Elderly residents are present.")
    if "insulin" in lowered or "diabetic" in lowered:
        facts.append("Medication-sensitive patients are present.")
    if water_match:
        facts.append(f"Water supply reported at about {water_match.group(1)} hours remaining.")
    if any(term in lowered for term in ["bridge", "blocked", "unsafe", "collapsed", "collapse", "debris", "isolate", "isolated", "route"]) or scenario_type == "damage":
        facts.append("Access route or infrastructure damage may affect response.")
    if any(term in lowered for term in ["fire", "flame", "smoke", "burning"]):
        facts.append("Fire or smoke hazard is reported or visually suspected.")
    if not facts:
        facts.append("Field report received; details require responder verification.")
    if image is not None:
        facts.append(f"Image received: {image.filename} ({image.content_type}, {image.size_bytes} bytes).")
    if before_image is not None:
        facts.append(f"Before image received for comparison: {before_image.filename}.")
    if image is not None and before_image is not None:
        facts.append("Before/after comparison requested; treat visible change assessment as requiring human verification when Ollama vision is offline.")

    urgency: Urgency = "medium"
    high_terms = ["insulin", "trapped", "collapse", "collapsed", "8 hours", "blocked", "critical", "unsafe", "debris", "isolate", "isolated", "fire", "flame", "smoke", "burning"]
    if any(term in lowered for term in high_terms) or (scenario_type == "damage" and any(term in lowered for term in ["bridge", "road", "route"])):
        urgency = "high"
    if any(term in lowered for term in ["unconscious", "life threat", "severe bleeding", "fire spreading"]):
        urgency = "critical"

    action_plan = build_action_plan(scenario_type, lowered, urgency)
    supply_request = build_supply_items(lowered, scenario_type)
    summary = build_summary(scenario_type, facts, urgency)
    sms_update = generate_sms_update(summary, urgency, location, action_plan, audience, role)
    recommendations = [action.next_step for action in action_plan]
    confidence = 0.62 if text else 0.48
    return AnalysisResult(
        scenario_type=scenario_type,
        engine="offline-fallback",
        model=model,
        location=location,
        summary=summary,
        urgency=urgency,
        extracted_facts=facts,
        inferred_recommendations=recommendations,
        action_plan=action_plan,
        supply_request=supply_request,
        sms_update=sms_update,
        citations=citations,
        confidence=confidence,
        verification_flags=safety_flags(text, scenario_type),
        image=image,
        before_image=before_image,
        role=role,
        sms_language=sms_language,
        localized_sms=localize_sms(sms_update, sms_language),
        tool_trace=build_tool_trace(bool(image), bool(before_image)),
        timeline=build_timeline("new", urgency),
    )


def build_action_plan(scenario_type: ScenarioType, lowered_text: str, urgency: Urgency) -> list[ActionItem]:
    actions: list[ActionItem] = []
    route_action = _route_action(urgency)
    has_route_risk = "bridge" in lowered_text or "blocked" in lowered_text or "unsafe" in lowered_text or "debris" in lowered_text or scenario_type == "damage"
    if "fire" in lowered_text or "flame" in lowered_text or "smoke" in lowered_text or "burning" in lowered_text:
        actions.append(
            ActionItem(
                priority="critical" if urgency == "critical" else "high",
                title="Escalate fire response",
                rationale="The report indicates a possible fire or smoke hazard.",
                owner="incident commander",
                next_step="Alert fire services, keep civilians away from the hazard area, and confirm evacuation or perimeter needs.",
            )
        )
    if scenario_type == "damage" and has_route_risk:
        actions.append(route_action)
    if "insulin" in lowered_text or "diabetic" in lowered_text or "medicine" in lowered_text:
        actions.append(
            ActionItem(
                priority="high",
                title="Protect medication-sensitive patients",
                rationale="The report mentions insulin or medication needs.",
                owner="medical liaison",
                next_step="Contact nearest clinic or ambulance coordinator for cold-chain medication support.",
            )
        )
    if "water" in lowered_text:
        actions.append(
            ActionItem(
                priority="high",
                title="Dispatch drinking water",
                rationale="The report indicates limited water supply.",
                owner="logistics lead",
                next_step="Send potable water and confirm distribution point at the shelter.",
            )
        )
    if scenario_type != "damage" and has_route_risk:
        actions.append(route_action)
    if "elderly" in lowered_text:
        actions.append(
            ActionItem(
                priority="medium",
                title="Check vulnerable residents",
                rationale="Older residents may need mobility, medication, or heat/cold support.",
                owner="shelter coordinator",
                next_step="Run a headcount and identify residents needing assisted transport or care.",
            )
        )
    if not actions:
        actions.append(
            ActionItem(
                priority=urgency,
                title="Verify field report",
                rationale="Initial report needs confirmation before resource dispatch.",
                owner="field coordinator",
                next_step="Call or radio the reporting contact and confirm location, hazards, and needs.",
            )
        )
    return actions


def _route_action(urgency: Urgency) -> ActionItem:
    return ActionItem(
        priority="high" if urgency != "critical" else "critical",
        title="Verify blocked access route",
        rationale="Route obstruction can delay medical and supply response.",
        owner="operations lead",
        next_step="Assign field team to confirm safe alternate access and mark blocked route.",
    )


def build_supply_items(lowered_text: str, scenario_type: ScenarioType) -> list[SupplyItem]:
    items: list[SupplyItem] = []
    if "water" in lowered_text:
        items.append(SupplyItem(item="Potable water", quantity="24-hour supply estimate", reason="Reported water shortage."))
    if "insulin" in lowered_text or "medicine" in lowered_text:
        items.append(SupplyItem(item="Cold-chain medication support", quantity="confirm patient count", reason="Medication-sensitive patients reported."))
    if "blanket" in lowered_text or "elderly" in lowered_text:
        items.append(SupplyItem(item="Blankets and comfort supplies", quantity="for vulnerable residents", reason="Shelter includes vulnerable residents."))
    if scenario_type == "damage" or "blocked" in lowered_text:
        items.append(SupplyItem(item="Route clearance assessment", quantity="one field crew", reason="Access route may be unsafe or blocked."))
    if "fire" in lowered_text or "flame" in lowered_text or "smoke" in lowered_text or "burning" in lowered_text:
        items.append(SupplyItem(item="Fire response support", quantity="dispatch through authority", reason="Possible fire or smoke hazard requires emergency response coordination."))
    return items


def build_summary(scenario_type: ScenarioType, facts: list[str], urgency: Urgency) -> str:
    label = "Shelter triage" if scenario_type == "shelter" else "Damage assessment" if scenario_type == "damage" else "Combined response"
    return f"{label} marked {urgency} based on: {' '.join(facts[:3])}"


def coerce_model_result(
    raw: dict,
    scenario_type: ScenarioType,
    location: str,
    audience: str,
    citations: list[GuidanceCitation],
    model: str,
    image: UploadedImageInfo | None = None,
    before_image: UploadedImageInfo | None = None,
    role: str = "district operations",
    sms_language: str = "English",
) -> AnalysisResult:
    urgency = raw.get("urgency", "medium")
    if urgency not in {"critical", "high", "medium", "low"}:
        urgency = "medium"
    actions = [
        ActionItem(
            priority=item.get("priority", urgency),
            title=item.get("title", "Field action"),
            rationale=item.get("rationale", "Model-recommended action."),
            owner=item.get("owner", "field coordinator"),
            next_step=item.get("next_step", item.get("next", "Verify and act through incident command.")),
        )
        for item in raw.get("action_plan", [])
        if isinstance(item, dict)
    ]
    supplies = [
        SupplyItem(
            item=item.get("item", "Resource"),
            quantity=item.get("quantity", "estimate required"),
            reason=item.get("reason", "Model-recommended resource."),
        )
        for item in raw.get("supply_request", [])
        if isinstance(item, dict)
    ]
    summary = raw.get("summary") or "Field report requires responder review."
    sms = raw.get("sms_update") or generate_sms_update(summary, urgency, location, actions, audience, role)
    combined_text = json.dumps(raw, ensure_ascii=True)
    flags = sorted(set(raw.get("verification_flags", []) + safety_flags(combined_text, scenario_type)))
    return AnalysisResult(
        scenario_type=scenario_type,
        engine="gemma4-ollama",
        model=model,
        location=location,
        summary=summary,
        urgency=urgency,
        extracted_facts=[str(item) for item in raw.get("extracted_facts", [])],
        inferred_recommendations=[str(item) for item in raw.get("inferred_recommendations", [])],
        action_plan=actions or build_action_plan(scenario_type, combined_text.lower(), urgency),
        supply_request=supplies or build_supply_items(combined_text.lower(), scenario_type),
        sms_update=sms,
        citations=citations,
        confidence=float(raw.get("confidence", 0.72)),
        verification_flags=flags,
        image=image,
        before_image=before_image,
        role=role,
        sms_language=sms_language,
        localized_sms=localize_sms(sms, sms_language),
        tool_trace=build_tool_trace(bool(image), bool(before_image)),
        timeline=build_timeline("new", urgency),
    )


def localize_sms(sms: str, language: str) -> str | None:
    if language == "English":
        return None
    prefixes = {
        "Hindi": "स्थानीय संदेश",
        "Tamil": "உள்ளூர் செய்தி",
        "Spanish": "Mensaje local",
    }
    prefix = prefixes.get(language, "Local message")
    return f"{prefix}: {sms}"


def build_tool_trace(has_image: bool, has_before_image: bool) -> list[str]:
    trace = ["retrieve_guidance", "analyze_field_input"]
    if has_before_image:
        trace.append("compare_before_after_images")
    elif has_image:
        trace.append("inspect_uploaded_image")
    trace.extend(["create_incident_report", "create_supply_request", "generate_sms_update", "save_local_incident"])
    return trace


def build_timeline(status: str, urgency: Urgency) -> list[TimelineEvent]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        TimelineEvent(label="Intake", detail="Field report captured locally.", timestamp=now),
        TimelineEvent(label="Prioritized", detail=f"Marked {urgency} by FieldAid.", timestamp=now),
        TimelineEvent(label="Status", detail=f"Current incident status: {status}.", timestamp=now),
    ]


def _combine_note_and_image_context(
    note_text: str,
    image: UploadedImageInfo | None,
    before_image: UploadedImageInfo | None = None,
) -> str:
    parts = [note_text.strip()]
    if before_image is not None:
        normalized_name = before_image.filename.replace("_", " ").replace("-", " ")
        parts.append(
            f"Before image file: {normalized_name}. Content type: {before_image.content_type}. "
            f"File size: {before_image.size_bytes} bytes."
        )
    if image is not None:
        normalized_name = image.filename.replace("_", " ").replace("-", " ")
        parts.append(
            f"Current image file: {normalized_name}. Content type: {image.content_type}. "
            f"File size: {image.size_bytes} bytes."
        )
    return "\n".join(part for part in parts if part).strip()
