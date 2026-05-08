from __future__ import annotations

import shutil
from dataclasses import dataclass
import os
from pathlib import Path
from uuid import uuid4

from app.image_classifier import classify_disaster_image
from app.models import FrameClassification, UploadedImageInfo, VideoClassifierAggregation, VideoDetection, VideoDisasterAnalysis, VideoScanResult
from app.ollama_client import call_gemma4_video_frames
from app.rag import retrieve_guidance
from app.tools import fallback_analysis


ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
MODELS_DIR = ROOT / "models"
RUNTIME_CACHE_DIR = Path(os.environ.get("FIELDAID_RUNTIME_CACHE", r"C:\hack\fieldaid_runtime"))
CUSTOM_MODEL_PATH = MODELS_DIR / "disaster_yolo.pt"
FALLBACK_MODEL_NAME = "yolo11n.pt"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_seconds: float
    path: Path


def validate_video_filename(filename: str) -> None:
    if Path(filename).suffix.lower() not in VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video type. Use one of: {allowed}.")


def compute_sample_indices(
    total_frames: int,
    fps: float,
    interval_seconds: float = 2.0,
    max_frames: int = 20,
) -> list[int]:
    if total_frames <= 0:
        return []
    safe_fps = fps if fps and fps > 0 else 1.0
    safe_interval = max(interval_seconds, 0.5)
    step = max(int(round(safe_fps * safe_interval)), 1)
    return list(range(0, total_frames, step))[:max_frames]


def select_model_source(models_dir: Path = MODELS_DIR) -> tuple[str, str]:
    custom_path = models_dir / "disaster_yolo.pt"
    if custom_path.exists():
        return str(custom_path), "custom"
    return FALLBACK_MODEL_NAME, "pretrained-fallback"


def sample_video_frames(
    video_path: Path,
    interval_seconds: float = 2.0,
    max_frames: int = 20,
    output_dir: Path = UPLOAD_DIR,
) -> list[SampledFrame]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for video frame sampling.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_indices = set(compute_sample_indices(total_frames, fps, interval_seconds, max_frames))
    if not target_indices:
        cap.release()
        raise ValueError("No readable frames found in uploaded video.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[SampledFrame] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in target_indices:
            path = output_dir / f"{uuid4().hex}_frame_{frame_index}.jpg"
            cv2.imwrite(str(path), frame)
            frames.append(SampledFrame(frame_index=frame_index, timestamp_seconds=frame_index / fps, path=path))
        frame_index += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError("No sampled frames could be written from uploaded video.")
    return frames


def run_yolo_on_frames(
    frames: list[SampledFrame],
    output_dir: Path = UPLOAD_DIR,
    confidence_threshold: float = 0.25,
) -> tuple[list[VideoDetection], list[UploadedImageInfo], str, str]:
    _prepare_yolo_runtime()
    model_ref, model_source = select_model_source()
    detections: list[VideoDetection] = []
    annotated_frames: list[UploadedImageInfo] = []
    try:
        from ultralytics import YOLO

        model = YOLO(model_ref)
        for sampled in frames:
            result = model.predict(str(sampled.path), conf=confidence_threshold, verbose=False)[0]
            names = result.names or {}
            for box in result.boxes or []:
                class_id = int(box.cls[0].item())
                class_name = str(names.get(class_id, class_id))
                detections.append(
                    VideoDetection(
                        frame_index=sampled.frame_index,
                        timestamp_seconds=round(sampled.timestamp_seconds, 2),
                        class_name=class_name,
                        confidence=round(float(box.conf[0].item()), 3),
                        bbox_xyxy=[round(float(value), 2) for value in box.xyxy[0].tolist()],
                    )
                )
            annotated_path = output_dir / f"{uuid4().hex}_annotated.jpg"
            plotted = result.plot()
            _write_image(annotated_path, plotted)
            annotated_frames.append(_image_info(annotated_path, "annotated YOLO frame"))
    except Exception:
        model_source = "heuristic-fallback"
        model_ref = "yolo-unavailable"
        for sampled in frames:
            annotated_path = output_dir / f"{uuid4().hex}_annotated.jpg"
            shutil.copyfile(sampled.path, annotated_path)
            annotated_frames.append(_image_info(annotated_path, "sampled video frame"))
    return detections, annotated_frames, model_source, model_ref


def _prepare_yolo_runtime() -> None:
    RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for child in ["ultralytics", "matplotlib", "torch", "tmp"]:
        (RUNTIME_CACHE_DIR / child).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_CACHE_DIR / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_DIR / "matplotlib"))
    os.environ.setdefault("TORCH_HOME", str(RUNTIME_CACHE_DIR / "torch"))
    os.environ.setdefault("TMP", str(RUNTIME_CACHE_DIR / "tmp"))
    os.environ.setdefault("TEMP", str(RUNTIME_CACHE_DIR / "tmp"))


def aggregate_video_observations(detections: list[VideoDetection]) -> list[str]:
    if not detections:
        return [
            "No YOLO detections met the configured threshold; manual review is required because absence of detections is not proof of safety.",
            "Treat video-derived route, bridge, road, or structural decisions as requiring human verification.",
        ]
    class_names = {detection.class_name.lower() for detection in detections}
    observations: list[str] = []
    if class_names & {"person"}:
        observations.append("People are visible in sampled frames; verify whether anyone is stranded, exposed, or blocking evacuation paths.")
    if class_names & {"car", "truck", "bus", "motorcycle", "bicycle", "boat"}:
        observations.append("Vehicles are visible in sampled frames; assess whether they obstruct access, evacuation, or supply delivery.")
    if class_names & {"traffic light", "stop sign"}:
        observations.append("Traffic control objects are visible; confirm route control and safe access status.")
    observations.append("Video detections are supporting evidence only; route or structural safety requires field verification.")
    return observations


def classify_sampled_frames(frames: list[SampledFrame]) -> VideoClassifierAggregation:
    frame_results: list[FrameClassification] = []
    label_counts: dict[str, int] = {}
    confidence_sums: dict[str, float] = {}
    unavailable_reason: str | None = None
    for frame in frames:
        result = classify_disaster_image(frame.path)
        if not result.get("available"):
            unavailable_reason = str(result.get("reason") or "classifier unavailable")
            continue
        label = result.get("top_label")
        confidence = float(result.get("top_confidence") or 0.0)
        if label:
            label_counts[str(label)] = label_counts.get(str(label), 0) + 1
            confidence_sums[str(label)] = confidence_sums.get(str(label), 0.0) + confidence
        frame_results.append(
            FrameClassification(
                frame_index=frame.frame_index,
                timestamp_seconds=round(frame.timestamp_seconds, 2),
                top_label=str(label) if label else None,
                top_confidence=confidence,
                confidence_band=str(result.get("confidence_band") or "uncertain"),
                predictions=list(result.get("predictions", [])),
            )
        )
    if not frame_results:
        return VideoClassifierAggregation(
            available=False,
            reason=unavailable_reason or "no frames were classified",
            observations=["Local MEDIC video-frame classifier was unavailable; use Gemma, YOLO, and manual review."],
        )

    top_label = max(label_counts, key=lambda label: (label_counts[label], confidence_sums[label] / max(label_counts[label], 1)))
    top_count = label_counts[top_label]
    avg_confidence = confidence_sums[top_label] / max(top_count, 1)
    frame_ratio = top_count / max(len(frame_results), 1)
    aggregate_confidence = min(0.95, (avg_confidence * 0.7) + (frame_ratio * 0.3))
    band = "likely" if aggregate_confidence >= 0.70 else "possible" if aggregate_confidence >= 0.45 else "uncertain"
    observations = [
        f"Local MEDIC classifier saw {top_label.replace('_', ' ')} in {top_count}/{len(frame_results)} sampled frames.",
        f"Average confidence for repeated {top_label.replace('_', ' ')} frames was {avg_confidence:.0%}.",
        "Frame classification is supporting evidence only; field command must verify life-safety, route, and structural decisions.",
    ]
    return VideoClassifierAggregation(
        available=True,
        model="mobilenet_v3_small MEDIC disaster classifier",
        frames_classified=len(frame_results),
        top_label=top_label,
        top_confidence=aggregate_confidence,
        confidence_band=band,
        label_counts=label_counts,
        frame_results=frame_results,
        observations=observations,
    )


async def analyze_disaster_from_frames(
    frames: list[SampledFrame],
    location: str,
    model: str,
) -> VideoDisasterAnalysis:
    try:
        raw = await call_gemma4_video_frames([frame.path for frame in frames], location, model)
        return coerce_video_disaster_analysis(raw, "gemma4-ollama")
    except Exception:
        return infer_disaster_from_frames(frames)


def coerce_video_disaster_analysis(raw: dict, engine: str) -> VideoDisasterAnalysis:
    disaster_type = str(raw.get("disaster_type", "unknown")).lower().strip() or "unknown"
    confidence = float(raw.get("confidence", 0.55))
    return VideoDisasterAnalysis(
        engine=engine,
        disaster_type=disaster_type,
        summary=str(raw.get("summary") or "Sampled video frames require field review."),
        confidence=max(0.0, min(confidence, 1.0)),
        visual_evidence=[str(item) for item in raw.get("visual_evidence", [])],
        recommended_focus=[str(item) for item in raw.get("recommended_focus", [])],
        verification_flags=[str(item) for item in raw.get("verification_flags", [])]
        or ["Video scene interpretation requires human verification."],
    )


def infer_disaster_from_frames(frames: list[SampledFrame]) -> VideoDisasterAnalysis:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return VideoDisasterAnalysis(
            engine="visual-heuristic-unavailable",
            disaster_type="unknown",
            summary="Gemma video analysis and visual fallback are unavailable; manual review is required.",
            confidence=0.3,
            verification_flags=["Manual review required because no video scene model was available."],
        )

    fire_scores: list[float] = []
    smoke_scores: list[float] = []
    for frame in frames:
        image = cv2.imread(str(frame.path))
        if image is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([0, 80, 120])
        upper_orange = np.array([35, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        fire_scores.append(float(cv2.countNonZero(fire_mask)) / float(image.shape[0] * image.shape[1]))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]
        smoke_mask = ((gray > 70) & (gray < 220) & (saturation < 70)).astype("uint8")
        smoke_scores.append(float(smoke_mask.sum()) / float(image.shape[0] * image.shape[1]))

    max_fire = max(fire_scores or [0.0])
    max_smoke = max(smoke_scores or [0.0])
    if max_fire > 0.025:
        return VideoDisasterAnalysis(
            engine="visual-heuristic",
            disaster_type="fire",
            summary="Sampled frames show strong orange/red flame-like regions consistent with a possible fire.",
            confidence=min(0.85, 0.45 + max_fire * 8),
            visual_evidence=[f"Flame-color pixel ratio peaked at {max_fire:.1%} across sampled frames."],
            recommended_focus=["fire response escalation", "civilian perimeter", "evacuation check", "medical standby"],
            verification_flags=["Fire identification from video frames requires human confirmation."],
        )
    if max_smoke > 0.18:
        return VideoDisasterAnalysis(
            engine="visual-heuristic",
            disaster_type="smoke",
            summary="Sampled frames contain broad low-saturation gray regions consistent with possible smoke or haze.",
            confidence=min(0.75, 0.35 + max_smoke),
            visual_evidence=[f"Smoke/haze heuristic ratio peaked at {max_smoke:.1%} across sampled frames."],
            recommended_focus=["air quality caution", "fire source check", "civilian perimeter"],
            verification_flags=["Smoke identification from video frames requires human confirmation."],
        )
    return VideoDisasterAnalysis(
        engine="visual-heuristic",
        disaster_type="unknown",
        summary="No clear disaster type was identified by Gemma or the local visual fallback.",
        confidence=0.35,
        visual_evidence=["No strong fire/smoke heuristic signal was found in sampled frames."],
        recommended_focus=["manual review", "field verification"],
        verification_flags=["Manual review required because the video scene is uncertain."],
    )


def merge_disaster_with_classifier(
    disaster_analysis: VideoDisasterAnalysis,
    classifier: VideoClassifierAggregation,
) -> VideoDisasterAnalysis:
    if not classifier.available or not classifier.top_label:
        return disaster_analysis
    if disaster_analysis.disaster_type not in {"unknown", "smoke"} and disaster_analysis.confidence >= classifier.top_confidence:
        disaster_analysis.visual_evidence.extend(classifier.observations)
        return disaster_analysis
    label = classifier.top_label
    focus_by_label = {
        "fire": ["fire response escalation", "evacuation check", "medical standby", "civilian perimeter"],
        "flood": ["floodwater exclusion", "alternate access routes", "water rescue readiness", "shelter supply continuity"],
        "road_damage": ["route closure verification", "alternate access", "responder traffic control"],
        "building_damage": ["structural verification", "search-and-rescue standby", "civilian exclusion zone"],
    }
    flags_by_label = {
        "fire": "Fire identification from classifier frames requires human confirmation.",
        "flood": "Flood identification from classifier frames requires human confirmation.",
        "road_damage": "Route damage from classifier frames requires human verification before travel decisions.",
        "building_damage": "Structural damage from classifier frames requires qualified human verification.",
    }
    return VideoDisasterAnalysis(
        engine=f"{disaster_analysis.engine}+medic-frame-classifier",
        disaster_type=label,
        summary=(
            f"Local MEDIC frame classifier identified likely {label.replace('_', ' ')} across sampled video frames. "
            f"{disaster_analysis.summary}"
        ),
        confidence=max(disaster_analysis.confidence, classifier.top_confidence),
        visual_evidence=[*classifier.observations, *disaster_analysis.visual_evidence],
        recommended_focus=focus_by_label.get(label, ["manual review", "field verification"]),
        verification_flags=[
            flags_by_label.get(label, "Classifier-derived video interpretation requires human confirmation."),
            *disaster_analysis.verification_flags,
        ],
    )


async def process_video_scan(
    video_path: Path,
    video_info: UploadedImageInfo,
    location: str,
    sample_interval_seconds: float,
    model: str,
    audience: str = "district emergency operations center",
) -> VideoScanResult:
    frames = sample_video_frames(video_path, sample_interval_seconds)
    disaster_analysis = await analyze_disaster_from_frames(frames, location, model)
    classifier_aggregation = classify_sampled_frames(frames)
    disaster_analysis = merge_disaster_with_classifier(disaster_analysis, classifier_aggregation)
    detections, annotated_frames, model_source, model_name = run_yolo_on_frames(frames)
    observations = aggregate_video_observations(detections)
    observations = [
        f"Gemma/visual disaster analysis identified: {disaster_analysis.disaster_type}.",
        disaster_analysis.summary,
        *classifier_aggregation.observations,
        *disaster_analysis.visual_evidence,
        *observations,
    ]
    note = (
        f"Video disaster analysis: likely {disaster_analysis.disaster_type}. "
        + disaster_analysis.summary
        + " "
        + " ".join(observations)
        + " Recommended response focus: "
        + "; ".join(disaster_analysis.recommended_focus)
        + f" Sampled {len(frames)} frames from uploaded video {video_info.filename}."
    )
    citations = retrieve_guidance(note, "damage")
    incident = fallback_analysis(
        "damage",
        note,
        location,
        audience,
        citations,
        model,
        image=annotated_frames[0] if annotated_frames else None,
    )
    incident.engine = f"video-gemma/{disaster_analysis.engine}+yolo/{model_source}"
    incident.tool_trace = [
        "save_video",
        "sample_frames",
        "analyze_frames_with_gemma",
        "classify_frames_with_medic_model",
        "run_yolo_supporting_detector",
        "aggregate_scene_and_detection_evidence",
        "create_incident_report",
        "save_local_incident",
    ]
    for flag in [
        *disaster_analysis.verification_flags,
        "Video-derived hazards require human verification.",
        "YOLO detections do not prove a route, bridge, or structure is safe.",
    ]:
        if flag not in incident.verification_flags:
            incident.verification_flags.append(flag)
    return VideoScanResult(
        video=video_info,
        model_source=model_source,
        model_name=model_name,
        sampled_frames=len(frames),
        disaster_analysis=disaster_analysis,
        classifier_aggregation=classifier_aggregation,
        detections=detections,
        annotated_frames=annotated_frames,
        observations=observations,
        incident=incident,
        tool_trace=incident.tool_trace,
    )


def _write_image(path: Path, image) -> None:
    import cv2

    cv2.imwrite(str(path), image)


def _image_info(path: Path, filename: str) -> UploadedImageInfo:
    return UploadedImageInfo(
        filename=filename,
        content_type="image/jpeg",
        size_bytes=path.stat().st_size,
        url=f"/uploads/{path.name}",
    )
