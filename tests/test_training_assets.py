import json
from pathlib import Path

from modeltraining.medic_qcri.download_medic import choose_indices, classify_fieldaid_label, output_for_label, safe_id
from training.build_vision_manifest import build_output, LABEL_TEMPLATES
from training.eval_utils import load_jsonl, score_predictions, validate_sft_record


ROOT = Path(__file__).resolve().parent.parent


def test_fieldaid_sft_seed_records_are_valid():
    records = load_jsonl(ROOT / "training" / "data" / "fieldaid_sft_seed.jsonl")
    assert len(records) >= 8
    for record in records:
        assert validate_sft_record(record) == []
        output_text = json.dumps(record["output"]).lower()
        assert "verification" in output_text
        assert "safe to" not in output_text


def test_fieldaid_eval_scoring_flags_json_and_safety():
    cases = load_jsonl(ROOT / "training" / "evals" / "fieldaid_eval.jsonl")
    predictions = []
    for case in cases:
        predictions.append(
            json.dumps(
                {
                    "summary": "Human verification required.",
                    "urgency": case["expected_urgency"],
                    "extracted_facts": case["expected_terms"],
                    "inferred_recommendations": ["Request human verification."],
                    "action_plan": [],
                    "supply_request": [],
                    "sms_update": "Verification required.",
                    "confidence": 0.7,
                    "verification_flags": ["Human verification required."],
                }
            )
        )
    score = score_predictions(cases, predictions)
    assert score.valid_json == len(cases)
    assert score.schema_ok == len(cases)
    assert score.urgency_ok == len(cases)
    assert score.forbidden_terms_ok == len(cases)


def test_vision_manifest_templates_produce_safe_fieldaid_outputs():
    for label, template in LABEL_TEMPLATES.items():
        output = build_output(template)
        output_text = json.dumps(output).lower()
        assert output["urgency"] in {"critical", "high", "medium", "low"}
        assert output["verification_flags"]
        assert "verification" in output_text or "review" in output_text
        if label != "unknown":
            assert "safe to" not in output_text


def test_medic_label_mapping_and_outputs_are_safe():
    assert classify_fieldaid_label({"event": "wildfire smoke"}) == "fire"
    assert classify_fieldaid_label({"event": "flood water road"}) == "flood"
    assert classify_fieldaid_label({"event": "collapsed building earthquake"}) == "building_damage"
    output = output_for_label("building_damage")
    assert output["urgency"] == "critical"
    assert output["verification_flags"]
    assert "safe" not in json.dumps(output).lower() or "do not" in json.dumps(output).lower()


def test_medic_safe_id_includes_index_to_avoid_overwrites():
    first = safe_id({"event_id": "wildfire"}, "train", 1)
    second = safe_id({"event_id": "wildfire"}, "train", 2)
    assert first != second
    assert first.startswith("train_000001")


def test_medic_fraction_sampling_is_deterministic():
    first = choose_indices(total=100, max_rows=0, fraction=0.25, seed=7)
    second = choose_indices(total=100, max_rows=0, fraction=0.25, seed=7)
    assert first == second
    assert len(first) == 25
    assert first == sorted(first)
