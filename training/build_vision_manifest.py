from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

LABEL_TEMPLATES = {
    "fire": {
        "urgency": "critical",
        "summary": "Image appears consistent with a possible active fire or smoke hazard.",
        "facts": ["Visible flame-like or smoke-like evidence in the uploaded image."],
        "recommendations": ["Escalate to fire response or incident command.", "Keep civilians clear until field authorities verify conditions."],
        "action_title": "Escalate possible fire",
        "action_next": "Request fire response and establish a civilian perimeter.",
        "supply_item": "Fire response and medical standby",
        "sms": "CRITICAL FieldAid update: possible fire/smoke in uploaded image. Keep civilians clear, request fire response, verify scene before action.",
        "flags": ["Image-derived fire status requires human confirmation.", "Do not declare the scene safe without field authority verification."],
    },
    "smoke": {
        "urgency": "high",
        "summary": "Image appears consistent with smoke or haze affecting air quality.",
        "facts": ["Visible smoke-like or haze-like evidence in the uploaded image."],
        "recommendations": ["Treat air quality as unverified.", "Check for a fire source and protect exposed civilians."],
        "action_title": "Verify smoke or haze source",
        "action_next": "Request source check and keep civilians clear of visible haze.",
        "supply_item": "Respiratory protection or medical standby",
        "sms": "HIGH FieldAid update: smoke/haze-like condition in uploaded image. Air quality unverified. Keep civilians clear and request source check.",
        "flags": ["Smoke or air-quality status requires human confirmation.", "Medical vulnerability decisions require qualified review."],
    },
    "flood": {
        "urgency": "high",
        "summary": "Image appears consistent with floodwater affecting access or safety.",
        "facts": ["Visible water or flood-like condition in the uploaded image."],
        "recommendations": ["Avoid routing people or vehicles through floodwater until verified.", "Request route assessment and alternate access planning."],
        "action_title": "Verify flood access risk",
        "action_next": "Request route verification and identify alternate access.",
        "supply_item": "Route control support",
        "sms": "HIGH FieldAid update: flood-like condition in uploaded image. Do not route through water until verified. Need route assessment and alternate access.",
        "flags": ["Flood route safety requires field verification.", "Do not infer passability from image alone."],
    },
    "road_damage": {
        "urgency": "high",
        "summary": "Image appears consistent with route or road damage affecting access.",
        "facts": ["Visible road or access damage evidence in the uploaded image."],
        "recommendations": ["Do not declare the route passable from image alone.", "Request public works or route authority assessment."],
        "action_title": "Verify damaged access route",
        "action_next": "Request route inspection and identify alternate access.",
        "supply_item": "Traffic control materials",
        "sms": "HIGH FieldAid update: road/access damage visible in uploaded image. Route status unverified. Need inspection and alternate access planning.",
        "flags": ["Route safety requires responsible authority verification.", "Image alone cannot prove passability."],
    },
    "building_damage": {
        "urgency": "critical",
        "summary": "Image appears consistent with possible structural or building damage.",
        "facts": ["Visible building or structural damage evidence in the uploaded image."],
        "recommendations": ["Keep civilians away from the structure until qualified inspection.", "Escalate to incident command or structural/public works authority."],
        "action_title": "Restrict unverified structure access",
        "action_next": "Establish perimeter and request qualified inspection.",
        "supply_item": "Perimeter control support",
        "sms": "CRITICAL FieldAid update: possible building/structural damage in uploaded image. Keep civilians clear. Need qualified inspection before entry.",
        "flags": ["Structural safety requires qualified human verification.", "Do not declare entry safe from image alone."],
    },
    "unknown": {
        "urgency": "medium",
        "summary": "Image does not provide enough clear evidence for a confident disaster classification.",
        "facts": ["Uploaded image evidence is unclear or ambiguous."],
        "recommendations": ["Request manual review and additional context.", "Avoid life-safety decisions based on unclear image evidence."],
        "action_title": "Request manual image review",
        "action_next": "Collect a clearer photo, location, and responder note.",
        "supply_item": "Manual review support",
        "sms": "MEDIUM FieldAid update: uploaded image is unclear. Manual review and additional context required before action.",
        "flags": ["Unclear image requires human review.", "Do not infer safety from uncertain visual evidence."],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FieldAid vision SFT manifest from labeled image folders.")
    parser.add_argument("--image-root", type=Path, required=True, help="Folder containing label subfolders such as fire/, flood/, road_damage/.")
    parser.add_argument("--output", type=Path, default=Path("training/data/fieldaid_vision_manifest.jsonl"))
    parser.add_argument("--location-prefix", default="Training image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for label, template in LABEL_TEMPLATES.items():
        folder = args.image_root / label
        if not folder.exists():
            continue
        for index, image_path in enumerate(sorted(folder.rglob("*")), start=1):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            record_id = f"{label}_{index:05d}"
            records.append(
                {
                    "id": record_id,
                    "image": str(image_path.as_posix()),
                    "scenario_type": "damage",
                    "location": f"{args.location_prefix}: {label.replace('_', ' ')}",
                    "instruction": "Inspect the uploaded image and return FieldAid disaster-response JSON. Do not declare any route, building, fire scene, or medical situation safe.",
                    "output": build_output(template),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


def build_output(template: dict) -> dict:
    return {
        "summary": template["summary"],
        "urgency": template["urgency"],
        "extracted_facts": template["facts"],
        "inferred_recommendations": template["recommendations"],
        "action_plan": [
            {
                "priority": template["urgency"],
                "title": template["action_title"],
                "rationale": template["summary"],
                "owner": "field coordinator",
                "next_step": template["action_next"],
            }
        ],
        "supply_request": [
            {
                "item": template["supply_item"],
                "quantity": "authority determined",
                "reason": template["summary"],
            }
        ],
        "sms_update": template["sms"],
        "confidence": 0.7,
        "verification_flags": template["flags"],
    }


if __name__ == "__main__":
    main()
