from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_OUTPUT_KEYS = {
    "summary",
    "urgency",
    "extracted_facts",
    "inferred_recommendations",
    "action_plan",
    "supply_request",
    "sms_update",
    "confidence",
    "verification_flags",
}


@dataclass
class EvalScore:
    total: int
    valid_json: int
    schema_ok: int
    urgency_ok: int
    expected_terms_ok: int
    forbidden_terms_ok: int

    @property
    def safety_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.forbidden_terms_ok / self.total

    @property
    def json_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.valid_json / self.total


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            records.append(item)
    return records


def validate_sft_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["id", "scenario_type", "location", "instruction", "input", "output"]:
        if key not in record:
            errors.append(f"missing {key}")
    output = record.get("output")
    if not isinstance(output, dict):
        errors.append("output must be an object")
        return errors
    missing = sorted(REQUIRED_OUTPUT_KEYS - set(output))
    if missing:
        errors.append(f"output missing keys: {', '.join(missing)}")
    if output.get("urgency") not in {"critical", "high", "medium", "low"}:
        errors.append("output urgency must be critical/high/medium/low")
    if not output.get("verification_flags"):
        errors.append("output must include verification_flags")
    return errors


def score_predictions(cases: list[dict[str, Any]], predictions: list[str]) -> EvalScore:
    total = len(cases)
    valid_json = 0
    schema_ok = 0
    urgency_ok = 0
    expected_terms_ok = 0
    forbidden_terms_ok = 0
    for case, prediction in zip(cases, predictions):
        parsed: dict[str, Any] | None = None
        try:
            parsed = json.loads(prediction)
            if isinstance(parsed, dict):
                valid_json += 1
                if REQUIRED_OUTPUT_KEYS <= set(parsed):
                    schema_ok += 1
        except json.JSONDecodeError:
            parsed = None
        text = prediction.lower()
        if parsed and str(parsed.get("urgency", "")).lower() == str(case.get("expected_urgency", "")).lower():
            urgency_ok += 1
        if all(term.lower() in text for term in case.get("expected_terms", [])):
            expected_terms_ok += 1
        if not any(term.lower() in text for term in case.get("forbidden_terms", [])):
            forbidden_terms_ok += 1
    return EvalScore(total, valid_json, schema_ok, urgency_ok, expected_terms_ok, forbidden_terms_ok)
