from pathlib import Path

import cv2
import numpy as np

from app.models import UploadedImageInfo, VideoDetection
from app.video_scan import (
    SampledFrame,
    aggregate_video_observations,
    classify_sampled_frames,
    coerce_video_disaster_analysis,
    compute_sample_indices,
    infer_disaster_from_frames,
    merge_disaster_with_classifier,
    select_model_source,
    validate_video_filename,
)


def test_compute_sample_indices_caps_and_intervals():
    indices = compute_sample_indices(total_frames=300, fps=30, interval_seconds=2, max_frames=3)
    assert indices == [0, 60, 120]


def test_select_model_source_prefers_custom_weights(tmp_path: Path):
    assert select_model_source(tmp_path)[1] == "pretrained-fallback"
    (tmp_path / "disaster_yolo.pt").write_bytes(b"weights")
    model_ref, source = select_model_source(tmp_path)
    assert source == "custom"
    assert model_ref.endswith("disaster_yolo.pt")


def test_aggregate_video_observations_never_declares_safe():
    observations = aggregate_video_observations([])
    assert any("manual review" in item.lower() for item in observations)
    detections = [
        VideoDetection(frame_index=0, timestamp_seconds=0.0, class_name="person", confidence=0.9),
        VideoDetection(frame_index=20, timestamp_seconds=2.0, class_name="truck", confidence=0.8),
    ]
    observations = aggregate_video_observations(detections)
    assert any("People are visible" in item for item in observations)
    assert any("Vehicles are visible" in item for item in observations)
    assert all("safe" not in item.lower() or "requires" in item.lower() or "not proof" in item.lower() for item in observations)


def test_validate_video_filename_rejects_unsupported_type():
    validate_video_filename("field-video.mp4")
    try:
        validate_video_filename("field-note.txt")
    except ValueError as exc:
        assert "Unsupported video type" in str(exc)
    else:
        raise AssertionError("Expected unsupported video type to fail")


def test_coerce_video_disaster_analysis_from_gemma_json():
    analysis = coerce_video_disaster_analysis(
        {
            "disaster_type": "fire",
            "summary": "Visible flames near a structure.",
            "confidence": 0.82,
            "visual_evidence": ["orange flames", "smoke plume"],
            "recommended_focus": ["evacuation", "fire response"],
        },
        "gemma4-ollama",
    )
    assert analysis.disaster_type == "fire"
    assert analysis.engine == "gemma4-ollama"
    assert "fire response" in analysis.recommended_focus


def test_visual_heuristic_identifies_fire(tmp_path: Path):
    frame_path = tmp_path / "fire_frame.jpg"
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:] = (20, 20, 20)
    cv2.rectangle(image, (20, 20), (140, 100), (0, 120, 255), -1)
    cv2.imwrite(str(frame_path), image)
    analysis = infer_disaster_from_frames([SampledFrame(frame_index=0, timestamp_seconds=0.0, path=frame_path)])
    assert analysis.disaster_type == "fire"
    assert analysis.confidence > 0.5


def test_frame_classifier_aggregation_counts_repeated_labels(monkeypatch, tmp_path: Path):
    frames = [
        SampledFrame(frame_index=0, timestamp_seconds=0.0, path=tmp_path / "a.jpg"),
        SampledFrame(frame_index=20, timestamp_seconds=2.0, path=tmp_path / "b.jpg"),
        SampledFrame(frame_index=40, timestamp_seconds=4.0, path=tmp_path / "c.jpg"),
    ]
    for frame in frames:
        frame.path.write_bytes(b"image")

    labels = ["fire", "fire", "flood"]

    def fake_classifier(_path):
        label = labels.pop(0)
        confidence = 0.8 if label == "fire" else 0.55
        return {
            "available": True,
            "model": "mobilenet_v3_small",
            "predictions": [{"label": label, "confidence": confidence}],
            "top_label": label,
            "top_confidence": confidence,
            "confidence_band": "likely",
        }

    monkeypatch.setattr("app.video_scan.classify_disaster_image", fake_classifier)
    aggregation = classify_sampled_frames(frames)
    assert aggregation.available is True
    assert aggregation.top_label == "fire"
    assert aggregation.label_counts["fire"] == 2
    assert any("2/3" in item for item in aggregation.observations)


def test_classifier_can_upgrade_unknown_video_analysis():
    analysis = coerce_video_disaster_analysis({"disaster_type": "unknown", "summary": "unclear"}, "gemma4-ollama")
    from app.models import VideoClassifierAggregation

    merged = merge_disaster_with_classifier(
        analysis,
        VideoClassifierAggregation(
            available=True,
            top_label="road_damage",
            top_confidence=0.72,
            confidence_band="likely",
            frames_classified=4,
            label_counts={"road_damage": 3},
            observations=["Local MEDIC classifier saw road damage in 3/4 sampled frames."],
        ),
    )
    assert merged.disaster_type == "road_damage"
    assert "medic-frame-classifier" in merged.engine
    assert any("Route damage" in item for item in merged.verification_flags)
