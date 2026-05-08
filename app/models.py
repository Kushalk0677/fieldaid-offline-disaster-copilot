from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ScenarioType = Literal["shelter", "damage", "combined"]
Urgency = Literal["critical", "high", "medium", "low"]


class GuidanceCitation(BaseModel):
    title: str
    source: str
    snippet: str
    score: float = 0.0


class ActionItem(BaseModel):
    priority: Urgency
    title: str
    rationale: str
    owner: str = "field coordinator"
    next_step: str


class SupplyItem(BaseModel):
    item: str
    quantity: str = "estimate required"
    reason: str


class UploadedImageInfo(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    url: str


class TimelineEvent(BaseModel):
    label: str
    detail: str
    timestamp: str | None = None


class VideoDetection(BaseModel):
    frame_index: int
    timestamp_seconds: float
    class_name: str
    confidence: float
    bbox_xyxy: list[float] = Field(default_factory=list)


class VideoDisasterAnalysis(BaseModel):
    engine: str
    disaster_type: str
    summary: str
    confidence: float = Field(ge=0, le=1, default=0.5)
    visual_evidence: list[str] = Field(default_factory=list)
    recommended_focus: list[str] = Field(default_factory=list)
    verification_flags: list[str] = Field(default_factory=list)


class FrameClassification(BaseModel):
    frame_index: int
    timestamp_seconds: float
    top_label: str | None = None
    top_confidence: float = Field(ge=0, le=1, default=0.0)
    confidence_band: str = "unavailable"
    predictions: list[dict] = Field(default_factory=list)


class VideoClassifierAggregation(BaseModel):
    available: bool = False
    model: str = "fieldaid-medic-image-classifier"
    frames_classified: int = 0
    top_label: str | None = None
    top_confidence: float = Field(ge=0, le=1, default=0.0)
    confidence_band: str = "unavailable"
    label_counts: dict[str, int] = Field(default_factory=dict)
    frame_results: list[FrameClassification] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    reason: str | None = None


class VideoScanResult(BaseModel):
    video: UploadedImageInfo
    model_source: str
    model_name: str
    sampled_frames: int
    disaster_analysis: VideoDisasterAnalysis | None = None
    classifier_aggregation: VideoClassifierAggregation | None = None
    detections: list[VideoDetection] = Field(default_factory=list)
    annotated_frames: list[UploadedImageInfo] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    incident: "AnalysisResult"
    tool_trace: list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    scenario_type: ScenarioType
    note_text: str = ""
    location: str = "Unknown location"
    audience: str = "district emergency operations center"
    model: str = "gemma4:e4b"


class AnalysisResult(BaseModel):
    scenario_type: ScenarioType
    engine: str
    model: str
    location: str
    summary: str
    urgency: Urgency
    extracted_facts: list[str] = Field(default_factory=list)
    inferred_recommendations: list[str] = Field(default_factory=list)
    action_plan: list[ActionItem] = Field(default_factory=list)
    supply_request: list[SupplyItem] = Field(default_factory=list)
    sms_update: str
    citations: list[GuidanceCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.65)
    verification_flags: list[str] = Field(default_factory=list)
    image: UploadedImageInfo | None = None
    before_image: UploadedImageInfo | None = None
    role: str = "district operations"
    sms_language: str = "English"
    localized_sms: str | None = None
    tool_trace: list[str] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    report_id: int | None = None
    created_at: str | None = None


class IncidentRecord(BaseModel):
    id: int
    created_at: str
    scenario_type: ScenarioType
    location: str
    urgency: Urgency
    summary: str
    status: str
    sms_update: str
    analysis_json: dict


class SyncQueueRecord(BaseModel):
    id: int
    created_at: str
    entity_type: str
    entity_id: int
    status: str
    payload_json: dict
    last_exported_at: str | None = None


class OutboundMessageRecord(BaseModel):
    id: int
    created_at: str
    incident_id: int | None = None
    channel: str
    destination: str
    message: str
    status: str
    metadata_json: dict = Field(default_factory=dict)


class OutboundCreateRequest(BaseModel):
    incident_id: int | None = None
    channel: str = "discord"
    destination: str
    message: str
    metadata_json: dict = Field(default_factory=dict)


class OutboundStatusUpdate(BaseModel):
    status: str


class SyncExportUpdate(BaseModel):
    ids: list[int] | None = None


class StatusUpdate(BaseModel):
    status: str


VideoScanResult.model_rebuild()
