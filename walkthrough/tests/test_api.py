import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.models import AnalysisResult, UploadedImageInfo, VideoScanResult


def test_index_loads():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "FieldAid" in response.text


def test_analyze_fallback_smoke():
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        data={
            "scenario_type": "shelter",
            "note_text": "43 people, insulin patients, water left for 8 hours, bridge blocked",
            "location": "Govt School",
            "model": "gemma4:e4b",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["report_id"] >= 1
    assert payload["urgency"] == "high"
    assert payload["citations"]


def test_analyze_image_only_preserves_upload():
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        data={
            "scenario_type": "damage",
            "note_text": "",
            "location": "Creek road",
            "model": "gemma4:e4b",
        },
        files={"image": ("Road_damaged_by_flood.jpg", b"fake-jpeg-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["urgency"] == "high"
    assert payload["image"]["filename"] == "Road_damaged_by_flood.jpg"
    assert payload["image"]["url"].startswith("/uploads/")
    assert "inspect_uploaded_image" in payload["tool_trace"]


def test_dashboard_grounding_and_exports():
    client = TestClient(app)
    challenge = client.post(
        "/api/grounding",
        data={"question": "Can civilians cross this damaged bridge?", "scenario_type": "damage"},
    )
    assert challenge.status_code == 200
    assert "human verification" in challenge.json()["answer"].lower()
    assert "discord_message" in challenge.json()
    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert "counts" in dashboard.json()
    sync = client.get("/api/sync/export")
    assert sync.status_code == 200
    assert sync.json()["export_type"] == "fieldaid_offline_sync_queue"


def test_handoff_packet_available_after_analysis():
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        data={
            "scenario_type": "damage",
            "note_text": "bridge route blocked",
            "location": "Bridge route",
            "model": "gemma4:e4b",
        },
    )
    report_id = response.json()["report_id"]
    packet = client.get(f"/api/incidents/{report_id}/handoff")
    assert packet.status_code == 200
    assert f"FieldAid Handoff Packet #{report_id}" in packet.text


def test_sync_queue_tracks_incident_and_export_status():
    client = TestClient(app)
    response = client.post(
        "/api/analyze",
        data={
            "scenario_type": "shelter",
            "note_text": "clinic shelter has medicine needs and low water",
            "location": "Clinic shelter",
            "model": "gemma4:e4b",
        },
    )
    report_id = response.json()["report_id"]

    queue = client.get("/api/sync/queue")
    assert queue.status_code == 200
    matching = [item for item in queue.json() if item["entity_type"] == "incident" and item["entity_id"] == report_id]
    assert matching
    assert matching[0]["status"] == "queued"

    export = client.get("/api/sync/export")
    assert export.status_code == 200
    assert export.json()["queue"]
    assert export.json()["exported_at"]

    marked = client.post("/api/sync/mark-exported", json={"ids": [matching[0]["id"]]})
    assert marked.status_code == 200
    exported = [item for item in marked.json() if item["id"] == matching[0]["id"]][0]
    assert exported["status"] == "exported"
    assert exported["last_exported_at"]


def test_discord_outbox_records_and_status_update():
    client = TestClient(app)
    created = client.post(
        "/api/outbox",
        json={
            "channel": "discord",
            "destination": "https://discord.com/channels/@me",
            "message": "**FieldAid status:** fire suspected. Human verification required.",
            "metadata_json": {"source": "test", "human_send_required": True},
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["status"] == "drafted"
    assert payload["metadata_json"]["human_send_required"] is True

    updated = client.patch(f"/api/outbox/{payload['id']}/status", json={"status": "opened"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "opened"

    outbox = client.get("/api/outbox")
    assert outbox.status_code == 200
    assert any(item["id"] == payload["id"] for item in outbox.json())


def test_grounding_accepts_photo_and_returns_discord_draft():
    client = TestClient(app)
    response = client.post(
        "/api/grounding",
        data={
            "question": "What is the status of this fire scene?",
            "scenario_type": "damage",
            "model": "gemma4:e4b",
        },
        files={"image": ("fire_scene.jpg", b"fake-jpeg", "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["image"]["filename"] == "fire_scene.jpg"
    assert payload["scene_type"] == "fire"
    assert payload["discord_message"].startswith("**FieldAid status:")
    assert "draft_discord_message" in payload["tool_trace"]


def test_grounding_photo_uses_local_visual_fallback_when_gemma_unavailable(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise RuntimeError("Unable to connect to Ollama")

    monkeypatch.setattr("app.grounding.call_gemma4_grounding_image", unavailable)
    monkeypatch.setattr(
        "app.grounding.classify_disaster_image",
        lambda *_args, **_kwargs: {
            "available": False,
            "model": "fieldaid-medic-image-classifier",
            "checkpoint": None,
            "classes": [],
            "predictions": [],
            "top_label": None,
            "top_confidence": 0.0,
            "confidence_band": "unavailable",
            "reason": "test disabled",
        },
    )
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:] = (15, 15, 15)
    cv2.rectangle(image, (20, 20), (140, 100), (0, 120, 255), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    client = TestClient(app)
    response = client.post(
        "/api/grounding",
        data={
            "question": "What is the status of this scene?",
            "scenario_type": "damage",
            "model": "gemma4:e4b",
        },
        files={"image": ("scene.jpg", encoded.tobytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scene_type"] == "fire"
    assert payload["status"] == "high"
    assert payload["engine"] == "gemma-unavailable/local-classifier-fallback"
    assert "local_visual_fallback" in payload["tool_trace"]
    assert "Ollama is not reachable" in payload["answer"]


def test_grounding_photo_uses_medic_classifier_when_gemma_unavailable(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise RuntimeError("Unable to connect to Ollama")

    monkeypatch.setattr("app.grounding.call_gemma4_grounding_image", unavailable)
    monkeypatch.setattr(
        "app.grounding.classify_disaster_image",
        lambda *_args, **_kwargs: {
            "available": True,
            "model": "mobilenet_v3_small",
            "checkpoint": "best.pt",
            "classes": ["building_damage", "fire", "flood", "road_damage"],
            "predictions": [
                {"label": "flood", "confidence": 0.72},
                {"label": "road_damage", "confidence": 0.18},
            ],
            "top_label": "flood",
            "top_confidence": 0.72,
            "confidence_band": "likely",
            "validation_accuracy": 0.6356,
        },
    )

    client = TestClient(app)
    response = client.post(
        "/api/grounding",
        data={
            "question": "What is the status of this scene?",
            "scenario_type": "damage",
            "model": "gemma4:e4b",
        },
        files={"image": ("scene.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["classifier"]["available"] is True
    assert payload["classifier"]["top_label"] == "flood"
    assert payload["scene_type"] == "flood"
    assert payload["status"] == "high"
    assert "run_medic_classifier" in payload["tool_trace"]
    assert "local MEDIC disaster classifier" in payload["answer"]


def test_video_scan_requires_video():
    client = TestClient(app)
    response = client.post("/api/video/scan", data={"location": "Creek road"})
    assert response.status_code == 400


def test_video_scan_mocked_creates_incident(monkeypatch):
    async def fake_process_video_scan(video_path, video_info, location, sample_interval_seconds, model):
        incident = AnalysisResult(
            scenario_type="damage",
            engine="video-yolo/heuristic-fallback",
            model=model,
            location=location,
            summary="Video scan marked high based on route hazard observations.",
            urgency="high",
            sms_update="HIGH FieldAid update: route hazard observed.",
            verification_flags=["Video-derived hazards require human verification."],
            tool_trace=["save_video", "sample_frames", "run_yolo", "create_incident_report"],
        )
        return VideoScanResult(
            video=video_info,
            model_source="heuristic-fallback",
            model_name="test-detector",
            sampled_frames=1,
            disaster_analysis={
                "engine": "gemma4-ollama",
                "disaster_type": "fire",
                "summary": "Visible flames in sampled frames.",
                "confidence": 0.8,
                "visual_evidence": ["flames"],
                "recommended_focus": ["fire response"],
                "verification_flags": ["Human verification required."],
            },
            detections=[],
            annotated_frames=[
                UploadedImageInfo(filename="frame.jpg", content_type="image/jpeg", size_bytes=10, url="/uploads/frame.jpg")
            ],
            observations=["Manual review required."],
            incident=incident,
            tool_trace=incident.tool_trace,
        )

    monkeypatch.setattr("app.main.process_video_scan", fake_process_video_scan)
    client = TestClient(app)
    response = client.post(
        "/api/video/scan",
        data={"location": "Creek road", "sample_interval_seconds": "2", "model": "gemma4:e4b"},
        files={"video": ("road.mp4", b"fake-video", "video/mp4")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["incident"]["report_id"] >= 1
    assert payload["disaster_analysis"]["disaster_type"] == "fire"
    assert payload["model_source"] == "heuristic-fallback"
    assert payload["tool_trace"]
