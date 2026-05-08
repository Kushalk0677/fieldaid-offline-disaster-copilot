from pathlib import Path

from app.db import list_incidents, save_incident, update_status
from app.models import GuidanceCitation, UploadedImageInfo
from app.rag import GuidanceDoc, retrieve_guidance
from app.tools import fallback_analysis, generate_sms_update, safety_flags


def test_retrieve_guidance_returns_relevant_citation():
    docs = [
        GuidanceDoc("Shelter Water", "x", "shelter", "Water supply and insulin medication support for elderly residents."),
        GuidanceDoc("Route Safety", "y", "damage", "Blocked bridges require authorized inspection."),
    ]
    result = retrieve_guidance("43 people insulin water", "shelter", docs=docs)
    assert result[0].title == "Shelter Water"
    assert "insulin" in result[0].snippet.lower()


def test_fallback_analysis_extracts_demo_facts():
    citations = [GuidanceCitation(title="Shelter Water", source="data/guidance/shelter_water.md", snippet="Water and medicine.")]
    result = fallback_analysis(
        "shelter",
        "43 people, 6 elderly, 2 insulin patients, water left for 8 hours, bridge blocked",
        "Govt School",
        "district EOC",
        citations,
        "gemma4:e4b",
    )
    assert result.urgency == "high"
    assert any("43" in fact for fact in result.extracted_facts)
    assert any(item.item == "Potable water" for item in result.supply_request)
    assert result.citations
    assert result.sms_update.startswith("HIGH")
    assert result.action_plan[0].title == "Protect medication-sensitive patients"


def test_fallback_analysis_uses_uploaded_image_context():
    image = UploadedImageInfo(
        filename="Road_damaged_by_flood.jpg",
        content_type="image/jpeg",
        size_bytes=71877,
        url="/uploads/demo.jpg",
    )
    result = fallback_analysis(
        "damage",
        "",
        "Creek road",
        "district EOC",
        [],
        "gemma4:e4b",
        image=image,
    )
    assert result.urgency == "high"
    assert result.image is not None
    assert result.image.filename == "Road_damaged_by_flood.jpg"
    assert any("Image received" in fact for fact in result.extracted_facts)


def test_safety_flags_include_human_verification():
    flags = safety_flags("insulin patient near damaged bridge", "damage")
    assert any("Medical" in flag for flag in flags)
    assert any("Structural" in flag for flag in flags)


def test_fire_action_precedes_route_action_for_video_damage():
    result = fallback_analysis(
        "damage",
        "Video disaster analysis: likely fire. Visible flames and smoke near a road access point.",
        "Warehouse road",
        "EOC",
        [],
        "gemma4:e4b",
    )
    assert result.action_plan[0].title == "Escalate fire response"


def test_sms_is_bounded():
    sms = generate_sms_update("x" * 500, "high", "Ward 7", [], "EOC", max_len=120)
    assert len(sms) <= 120


def test_sqlite_incident_lifecycle(tmp_path: Path):
    db_path = tmp_path / "fieldaid.sqlite"
    result = fallback_analysis("damage", "blocked bridge", "Market bridge", "EOC", [], "gemma4:e4b")
    saved = save_incident(result, db_path=db_path)
    assert saved.report_id == 1
    records = list_incidents(db_path=db_path)
    assert records[0].location == "Market bridge"
    updated = update_status(1, "dispatched", db_path=db_path)
    assert updated is not None
    assert updated.status == "dispatched"
