from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm


DATASET_NAME = "QCRI/MEDIC"
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
IMAGE_DIR = ROOT / "images"
FIELDAID_DIR = ROOT / "fieldaid"
FIELDAID_IMAGE_DIR = FIELDAID_DIR / "images"


LABEL_FOLDERS = {
    "fire": ["fire", "wildfire", "smoke", "burn", "burning"],
    "flood": ["flood", "water", "hurricane", "storm", "typhoon"],
    "road_damage": ["road", "bridge", "transport", "infrastructure", "vehicle", "accident"],
    "building_damage": ["building", "collapsed", "collapse", "rubble", "earthquake", "damage", "destroyed"],
    "affected_people": ["affected", "injured", "rescue", "evacuat", "people", "shelter"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download QCRI/MEDIC and prepare FieldAid vision training assets.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "hf_cache")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional per-split row limit for smoke tests.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Deterministic fraction of each split to convert, e.g. 0.25.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean-output", action="store_true", help="Delete raw/images/fieldaid outputs before conversion; keeps Hugging Face cache.")
    parser.add_argument("--link-fieldaid-images", action="store_true", help="Hardlink FieldAid label images to split images instead of copying when possible.")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--skip-fieldaid-copy", action="store_true", help="Only save raw split images/metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Run: python -m pip install -r modeltraining/medic_qcri/requirements.txt"
        ) from exc

    if args.clean_output:
        for directory in [RAW_DIR, IMAGE_DIR, FIELDAID_DIR]:
            if directory.exists():
                shutil.rmtree(directory)

    for directory in [RAW_DIR, IMAGE_DIR, FIELDAID_DIR, FIELDAID_IMAGE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    if not args.no_preflight:
        preflight_huggingface_access()

    print(f"Loading {args.dataset} from Hugging Face...")
    try:
        dataset = load_dataset(args.dataset, cache_dir=str(args.cache_dir))
    except Exception as exc:
        raise SystemExit(
            "Could not download QCRI/MEDIC from Hugging Face.\n"
            f"Reason: {type(exc).__name__}: {exc}\n\n"
            "If you are in the Codex sandbox, rerun with network approval. "
            "If local Windows networking keeps failing, run this script on your SSH GPU machine instead:\n"
            "  python modeltraining/medic_qcri/download_medic.py --max-rows 25\n"
            "Then remove --max-rows for the full download."
        ) from exc
    manifest_path = FIELDAID_DIR / "fieldaid_vision_manifest.jsonl"
    skipped_path = RAW_DIR / "skipped_rows.jsonl"
    label_counts: Counter[str] = Counter()
    fieldaid_rows = 0
    skipped_rows = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_handle, skipped_path.open("w", encoding="utf-8") as skipped_handle:
        for split_name, split in dataset.items():
            split_dir = IMAGE_DIR / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = RAW_DIR / f"metadata_{split_name}.jsonl"
            indices = choose_indices(len(split), args.max_rows, args.fraction, args.seed + len(split_name))
            print(f"Saving {len(indices)} rows from split '{split_name}'...")
            with metadata_path.open("w", encoding="utf-8") as metadata_handle:
                for index in tqdm(indices, desc=split_name):
                    try:
                        row = split[index]
                        image = extract_image(row)
                        image_id = safe_id(row, split_name, index)
                        image_filename = f"{image_id}.{args.image_format}"
                        image_path = split_dir / image_filename
                        if args.overwrite or not image_path.exists():
                            save_image(image, image_path, args.image_format)

                        metadata = metadata_from_row(row, split_name, index, image_path)
                        metadata_handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

                        if not args.skip_fieldaid_copy:
                            label = classify_fieldaid_label(metadata)
                            label_counts[label] += 1
                            target_dir = FIELDAID_IMAGE_DIR / label
                            target_dir.mkdir(parents=True, exist_ok=True)
                            target_image = target_dir / image_filename
                            if args.overwrite or not target_image.exists():
                                copy_or_link_image(image_path, target_image, args.link_fieldaid_images)
                            manifest_handle.write(
                                json.dumps(build_fieldaid_record(metadata, target_image, label), ensure_ascii=False) + "\n"
                            )
                            fieldaid_rows += 1
                    except Exception as exc:
                        skipped_rows += 1
                        skipped_handle.write(
                            json.dumps(
                                {
                                    "split": split_name,
                                    "index": index,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

    summary = {
        "dataset": args.dataset,
        "max_rows_per_split": args.max_rows or None,
        "fraction": args.fraction,
        "fieldaid_rows": fieldaid_rows,
        "skipped_rows": skipped_rows,
        "label_counts": dict(label_counts),
    }
    (FIELDAID_DIR / "label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def choose_indices(total: int, max_rows: int, fraction: float, seed: int) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and less than or equal to 1.")
    target_count = total
    if fraction < 1:
        target_count = max(1, math.floor(total * fraction))
    if max_rows:
        target_count = min(target_count, max_rows)
    if target_count >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), target_count))


def copy_or_link_image(source: Path, target: Path, link: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            if target.exists():
                target.unlink()
            target.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def extract_image(row: dict[str, Any]) -> Image.Image:
    image = row.get("image")
    if image is None:
        for value in row.values():
            if isinstance(value, Image.Image):
                image = value
                break
    if not isinstance(image, Image.Image):
        raise ValueError("Could not find a PIL image column in MEDIC row.")
    return image.convert("RGB")


def preflight_huggingface_access() -> None:
    request = Request("https://huggingface.co/datasets/QCRI/MEDIC", method="HEAD")
    try:
        with urlopen(request, timeout=15) as response:
            if response.status >= 400:
                raise RuntimeError(f"Hugging Face returned HTTP {response.status}")
    except (OSError, URLError, RuntimeError) as exc:
        print(
            "Hugging Face preflight failed. Network access is required to download MEDIC.\n"
            f"Reason: {type(exc).__name__}: {exc}\n"
            "Tip: run with network approval here, or run this downloader directly on your SSH GPU machine.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def save_image(image: Image.Image, path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpg":
        image.save(path, format="JPEG", quality=92, optimize=True)
    else:
        image.save(path, format="PNG", optimize=True)


def safe_id(row: dict[str, Any], split_name: str, index: int) -> str:
    for key in ["image_id", "id", "tweet_id", "event_id"]:
        value = row.get(key)
        if value not in [None, ""]:
            return slugify(f"{split_name}_{index:06d}_{value}")
    return f"{split_name}_{index:06d}"


def metadata_from_row(row: dict[str, Any], split_name: str, index: int, image_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "split": split_name,
        "index": index,
        "image_path": image_path.as_posix(),
    }
    for key, value in row.items():
        if isinstance(value, Image.Image):
            continue
        metadata[key] = normalize_value(value)
    return metadata


def normalize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    return str(value)


def classify_fieldaid_label(metadata: dict[str, Any]) -> str:
    text = json.dumps(metadata, ensure_ascii=False).lower()
    for label, terms in LABEL_FOLDERS.items():
        if any(term in text for term in terms):
            return label
    return "unknown"


def build_fieldaid_record(metadata: dict[str, Any], image_path: Path, label: str) -> dict[str, Any]:
    output = output_for_label(label)
    return {
        "id": slugify(f"medic_{metadata['split']}_{metadata['index']}_{label}"),
        "image": image_path.as_posix(),
        "source_dataset": DATASET_NAME,
        "source_metadata": metadata,
        "scenario_type": "damage",
        "location": f"MEDIC {label.replace('_', ' ')} sample",
        "instruction": "Inspect the uploaded disaster image and return FieldAid JSON. Do not declare any road, bridge, building, fire scene, or medical situation safe.",
        "output": output,
    }


def output_for_label(label: str) -> dict[str, Any]:
    templates = {
        "fire": (
            "critical",
            "Image appears consistent with a possible fire or smoke hazard.",
            "Escalate possible fire",
            "Request fire response or incident command review and establish a civilian perimeter.",
            "CRITICAL FieldAid update: possible fire/smoke in MEDIC image. Keep civilians clear, request response, verify before action.",
            ["Image-derived fire status requires human confirmation.", "Do not declare the scene safe without field authority verification."],
        ),
        "flood": (
            "high",
            "Image appears consistent with floodwater or storm-related access risk.",
            "Verify flood access risk",
            "Request route verification and identify alternate access before movement.",
            "HIGH FieldAid update: possible flood/access hazard in MEDIC image. Route status unverified; request assessment.",
            ["Flood route safety requires field verification.", "Do not infer passability from image alone."],
        ),
        "road_damage": (
            "high",
            "Image appears consistent with possible road, bridge, vehicle, or access disruption.",
            "Verify access route",
            "Request route authority review and identify alternate access.",
            "HIGH FieldAid update: possible route/access disruption in MEDIC image. Need verification and alternate access planning.",
            ["Route safety requires responsible authority verification.", "Image alone cannot prove passability."],
        ),
        "building_damage": (
            "critical",
            "Image appears consistent with possible building or structural damage.",
            "Restrict unverified structure access",
            "Keep civilians away and request qualified structural or public works inspection.",
            "CRITICAL FieldAid update: possible structural damage in MEDIC image. Keep civilians clear and request qualified inspection.",
            ["Structural safety requires qualified human verification.", "Do not declare entry safe from image alone."],
        ),
        "affected_people": (
            "high",
            "Image appears to involve affected people, sheltering, rescue, or evacuation context.",
            "Verify people and medical needs",
            "Request responder review for injuries, shelter needs, evacuation, and access constraints.",
            "HIGH FieldAid update: MEDIC image may involve affected people. Verify needs, safety, medical, and evacuation status.",
            ["Medical and evacuation decisions require qualified human verification.", "Image alone may not reveal all vulnerabilities."],
        ),
        "unknown": (
            "medium",
            "Image is disaster-related or ambiguous but does not map cleanly to a FieldAid hazard class.",
            "Request manual image review",
            "Collect clearer context, location, and responder notes before life-safety action.",
            "MEDIUM FieldAid update: MEDIC image is ambiguous for FieldAid class. Manual review and more context required.",
            ["Unclear image requires human review.", "Do not infer safety from uncertain visual evidence."],
        ),
    }
    urgency, summary, action_title, next_step, sms, flags = templates[label]
    return {
        "summary": summary,
        "urgency": urgency,
        "extracted_facts": [summary],
        "inferred_recommendations": [next_step, "Maintain human verification for life-safety decisions."],
        "action_plan": [
            {
                "priority": urgency,
                "title": action_title,
                "rationale": summary,
                "owner": "field coordinator",
                "next_step": next_step,
            }
        ],
        "supply_request": [
            {
                "item": "Field verification support",
                "quantity": "authority determined",
                "reason": summary,
            }
        ],
        "sms_update": sms,
        "confidence": 0.68,
        "verification_flags": flags,
    }


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return value.strip("_")[:140] or "medic_row"


if __name__ == "__main__":
    main()
