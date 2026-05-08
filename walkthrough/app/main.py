from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.db import (
    create_outbound_message,
    init_db,
    list_incidents,
    list_outbound_messages,
    list_sync_queue,
    mark_sync_exported,
    save_incident,
    update_outbound_status,
    update_status,
)
from app.grounding import analyze_grounding
from app.models import (
    AnalysisResult,
    IncidentRecord,
    OutboundCreateRequest,
    OutboundMessageRecord,
    OutboundStatusUpdate,
    ScenarioType,
    StatusUpdate,
    SyncExportUpdate,
    SyncQueueRecord,
    UploadedImageInfo,
    VideoScanResult,
)
from app.ollama_client import call_gemma4
from app.rag import retrieve_guidance
from app.tools import coerce_model_result, fallback_analysis, safety_flags
from app.video_scan import process_video_scan, validate_video_filename


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WALKTHROUGH_MEDIA_DIR = ROOT / "media"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="FieldAid", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
if WALKTHROUGH_MEDIA_DIR.exists():
    shutil_app = True
    app.mount("/wt-media", StaticFiles(directory=str(WALKTHROUGH_MEDIA_DIR)), name="wt-media")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/walkthrough/steps")
def walkthrough_steps():
    return [
        {"id": 1, "title": "Shelter Intake", "page": "intake", "tab": "shelter", "note": "43 people in the school building. 6 elderly residents, 2 insulin patients need refrigeration. Water left for about 8 hours. Bridge to main road is blocked.", "location": "Government School, Sector 7", "model": "gemma4:e4b", "media": None, "button": "#analyzeShelter", "question": None, "video_type": None, "sms_language": "English", "role": "district operations"},
        {"id": 2, "title": "Flood Damage Assessment", "page": "intake", "tab": "damage", "note": "Creek road bridge completely down. Vehicles cannot pass. Alternative route through Hilltown but road washed out at two points.", "location": "Creek Road Bridge, Sector 7", "model": "gemma4:e4b", "media": "flooded_highway.jpg", "button": "#assessDamage", "question": None, "video_type": None, "sms_language": "English", "role": "district operations"},
        {"id": 3, "title": "Fire Video Scan", "page": "video", "tab": None, "note": None, "location": "Sector 7 wildfire area", "model": "gemma4:e4b", "media": "wildfire.mp4", "button": "#runVideoScan", "question": None, "video_type": "mp4", "sms_language": "English", "role": "district operations"},
        {"id": 4, "title": "Hurricane Video Scan", "page": "video", "tab": None, "note": None, "location": "Coastal route affected by hurricane", "model": "gemma4:e4b", "media": "hurricane.mp4", "button": "#runVideoScan", "question": None, "video_type": "mp4", "sms_language": "English", "role": "district operations"},
        {"id": 5, "title": "Trust Gate: Is fire area safe?", "page": "trust", "tab": None, "note": None, "location": None, "model": "gemma4:e4b", "media": "wildfire_near_road.jpg", "button": "#groundingButton", "question": "Is this area safe for civilian access near the fire?", "video_type": None, "sms_language": "English", "role": None},
        {"id": 6, "title": "Trust Gate: Can we cross the flooded highway?", "page": "trust", "tab": None, "note": None, "location": None, "model": "gemma4:e4b", "media": "flooded_highway.jpg", "button": "#groundingButton", "question": "Can civilians safely cross this flooded highway?", "video_type": None, "sms_language": "English", "role": None},
        {"id": 7, "title": "Earthquake Video Scan", "page": "video", "tab": None, "note": None, "location": "Earthquake affected zone", "model": "gemma4:e4b", "media": "earthquake.mp4", "button": "#runVideoScan", "question": None, "video_type": "mp4", "sms_language": "English", "role": "district operations"},
        {"id": 8, "title": "Operations Dashboard", "page": "dashboard", "tab": None, "note": None, "location": None, "model": "gemma4:e4b", "media": None, "button": "#refreshDashboard", "question": None, "video_type": None, "sms_language": "English", "role": None},
        {"id": 9, "title": "Incident Log", "page": "incidents", "tab": None, "note": None, "location": None, "model": "gemma4:e4b", "media": None, "button": "#refreshLog", "question": None, "video_type": None, "sms_language": "English", "role": None},
        {"id": 10, "title": "Discord Handoff Draft", "page": "exports", "tab": None, "note": None, "location": None, "model": "gemma4:e4b", "media": None, "button": "#exportSync", "question": None, "video_type": None, "sms_language": "English", "role": None},
        {"id": 11, "title": "Offline Sync Export", "page": "exports", "tab": None, "note": None, "location": None, "model": "gemma4:e4b", "media": None, "button": "#markSyncExported", "question": None, "video_type": None, "sms_language": "English", "role": None},
    ]


@app.get("/api/walkthrough/media/{filename}")
def walkthrough_media(filename: str):
    media_path = WALKTHROUGH_MEDIA_DIR / filename
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(media_path)


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(
    scenario_type: ScenarioType = Form(...),
    note_text: str = Form(""),
    location: str = Form("Unknown location"),
    audience: str = Form("district emergency operations center"),
    model: str = Form("gemma4:e4b"),
    role: str = Form("district operations"),
    sms_language: str = Form("English"),
    image: UploadFile | None = File(None),
    before_image: UploadFile | None = File(None),
) -> AnalysisResult:
    image_info, image_path = await _save_upload(image) if image and image.filename else (None, None)
    before_image_info, before_image_path = await _save_upload(before_image) if before_image and before_image.filename else (None, None)
    query = " ".join(
        [
            scenario_type,
            location,
            note_text,
            image_info.filename if image_info else "",
            before_image_info.filename if before_image_info else "",
        ]
    )
    citations = retrieve_guidance(query, scenario_type)
    guidance_context = "\n\n".join(f"{c.title}: {c.snippet}" for c in citations)
    try:
        raw = await call_gemma4(
            scenario_type=scenario_type,
            note_text=note_text,
            location=location,
            guidance_context=guidance_context,
            model=model,
            image_path=image_path,
            before_image_path=before_image_path,
        )
        result = coerce_model_result(raw, scenario_type, location, audience, citations, model, image_info, before_image_info, role, sms_language)
    except Exception:
        result = fallback_analysis(scenario_type, note_text, location, audience, citations, model, image_info, before_image_info, role, sms_language)
    return save_incident(result)


@app.post("/api/video/scan", response_model=VideoScanResult)
async def video_scan(
    video: UploadFile | None = File(None),
    location: str = Form("Unknown video location"),
    sample_interval_seconds: float = Form(2.0),
    model: str = Form("gemma4:e4b"),
) -> VideoScanResult:
    if video is None or not video.filename:
        raise HTTPException(status_code=400, detail="Upload a video file to scan.")
    try:
        validate_video_filename(video.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    video_info, video_path = await _save_upload(video)
    if video_info is None or video_path is None:
        raise HTTPException(status_code=400, detail="Video upload could not be saved.")
    try:
        scan = await process_video_scan(
            video_path=video_path,
            video_info=video_info,
            location=location,
            sample_interval_seconds=sample_interval_seconds,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    scan.incident = save_incident(scan.incident)
    return scan


@app.get("/api/incidents", response_model=list[IncidentRecord])
def incidents() -> list[IncidentRecord]:
    return list_incidents()


@app.get("/api/dashboard")
def dashboard() -> dict:
    records = list_incidents()
    categories = {
        "urgent_medical": 0,
        "water_shortage": 0,
        "blocked_routes": 0,
        "verification_flags": 0,
        "waiting_sync": len(records),
    }
    routes: list[dict] = []
    for record in records:
        payload = record.analysis_json
        text = " ".join(
            [
                record.summary,
                record.sms_update,
                " ".join(payload.get("extracted_facts", [])),
                " ".join(payload.get("verification_flags", [])),
            ]
        ).lower()
        if any(term in text for term in ["insulin", "medical", "medicine", "patient"]):
            categories["urgent_medical"] += 1
        if "water" in text:
            categories["water_shortage"] += 1
        if any(term in text for term in ["blocked", "bridge", "route", "road", "access"]):
            categories["blocked_routes"] += 1
            routes.append({"id": record.id, "location": record.location, "urgency": record.urgency, "status": record.status})
        if payload.get("verification_flags"):
            categories["verification_flags"] += 1
    return {"counts": categories, "routes": routes[:8], "total": len(records)}


@app.get("/api/sync/export")
def sync_export() -> JSONResponse:
    queue = list_sync_queue()
    return JSONResponse(
        {
            "export_type": "fieldaid_offline_sync_queue",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "queue": [record.model_dump(mode="json") for record in queue],
            "incidents": [record.model_dump(mode="json") for record in list_incidents()],
            "note": "Export does not delete local records. Use mark-exported after transfer is confirmed.",
        }
    )


@app.get("/api/sync/queue", response_model=list[SyncQueueRecord])
def sync_queue(status: str | None = None) -> list[SyncQueueRecord]:
    return list_sync_queue(status=status)


@app.post("/api/sync/mark-exported", response_model=list[SyncQueueRecord])
def sync_mark_exported(update: SyncExportUpdate | None = None) -> list[SyncQueueRecord]:
    return mark_sync_exported(update.ids if update else None)


@app.get("/api/outbox", response_model=list[OutboundMessageRecord])
def outbound_messages() -> list[OutboundMessageRecord]:
    return list_outbound_messages()


@app.post("/api/outbox", response_model=OutboundMessageRecord)
def outbound_create(request: OutboundCreateRequest) -> OutboundMessageRecord:
    if request.channel.strip().lower() != "discord":
        raise HTTPException(status_code=400, detail="Only Discord outbox handoff is supported in this MVP.")
    if not request.destination.strip():
        raise HTTPException(status_code=400, detail="Discord destination is required for auditable handoff.")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Outbound message cannot be empty.")
    return create_outbound_message(
        channel=request.channel.strip().lower(),
        destination=request.destination.strip(),
        message=request.message.strip(),
        incident_id=request.incident_id,
        metadata_json=request.metadata_json,
    )


@app.patch("/api/outbox/{message_id}/status", response_model=OutboundMessageRecord)
def outbound_set_status(message_id: int, update: OutboundStatusUpdate) -> OutboundMessageRecord:
    record = update_outbound_status(message_id, update.status)
    if record is None:
        raise HTTPException(status_code=404, detail="Outbox message not found")
    return record


@app.get("/api/incidents/{report_id}/handoff")
def handoff_packet(report_id: int) -> PlainTextResponse:
    record = next((item for item in list_incidents() if item.id == report_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    payload = record.analysis_json
    lines = [
        f"FieldAid Handoff Packet #{record.id}",
        f"Location: {record.location}",
        f"Urgency: {record.urgency}",
        f"Status: {record.status}",
        f"Created: {record.created_at}",
        "",
        "Summary:",
        record.summary,
        "",
        "SMS / Radio:",
        record.sms_update,
        "",
        "Actions:",
    ]
    for action in payload.get("action_plan", []):
        lines.append(f"- [{action.get('priority')}] {action.get('title')}: {action.get('next_step')}")
    lines.extend(["", "Verification Flags:"])
    for flag in payload.get("verification_flags", []):
        lines.append(f"- {flag}")
    lines.extend(["", "Citations:"])
    for citation in payload.get("citations", []):
        lines.append(f"- {citation.get('title')} ({citation.get('source')})")
    return PlainTextResponse("\n".join(lines))


@app.post("/api/grounding")
async def grounding_challenge(
    question: str = Form(...),
    scenario_type: ScenarioType = Form("damage"),
    model: str = Form("gemma4:e4b"),
    image: UploadFile | None = File(None),
) -> dict:
    image_info, image_path = await _save_upload(image) if image and image.filename else (None, None)
    return await analyze_grounding(question, scenario_type, image_info, image_path, model)


@app.patch("/api/incidents/{report_id}/status", response_model=IncidentRecord)
def set_status(report_id: int, update: StatusUpdate) -> IncidentRecord:
    record = update_status(report_id, update.status)
    if record is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return record


async def _save_upload(upload: UploadFile | None) -> tuple[UploadedImageInfo, Path] | tuple[None, None]:
    if upload is None or not upload.filename:
        return None, None
    suffix = Path(upload.filename).suffix.lower() or ".upload"
    target = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    content = await upload.read()
    target.write_bytes(content)
    info = UploadedImageInfo(
        filename=upload.filename,
        content_type=upload.content_type or "application/octet-stream",
        size_bytes=len(content),
        url=f"/uploads/{target.name}",
    )
    return info, target
