from __future__ import annotations

import argparse
from pathlib import Path

from training.eval_utils import load_jsonl, validate_sft_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FieldAid training/eval JSONL assets.")
    parser.add_argument("--data", type=Path, default=Path("training/data/fieldaid_sft_seed.jsonl"))
    parser.add_argument("--evals", type=Path, default=Path("training/evals/fieldaid_eval.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    records = load_jsonl(args.data)
    for record in records:
        for error in validate_sft_record(record):
            errors.append(f"{record.get('id', '<unknown>')}: {error}")
    evals = load_jsonl(args.evals)
    for case in evals:
        for key in ["id", "scenario_type", "location", "input", "expected_urgency", "expected_terms", "forbidden_terms"]:
            if key not in case:
                errors.append(f"{case.get('id', '<unknown eval>')}: missing {key}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(records)} SFT records and {len(evals)} eval cases.")


if __name__ == "__main__":
    main()
