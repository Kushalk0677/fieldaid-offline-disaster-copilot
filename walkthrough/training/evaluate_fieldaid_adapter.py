from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.eval_utils import load_jsonl, score_predictions
from training.train_fieldaid_lora import SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FieldAid safety/JSON benchmarks for a Gemma adapter.")
    parser.add_argument("--model-name", default="models/fieldaid-gemma4-e4b-lora", help="Adapter path or merged model path.")
    parser.add_argument("--base-model", default="google/gemma-4-E4B-it", help="Base model if loading a LoRA adapter.")
    parser.add_argument("--evals", type=Path, default=Path("training/evals/fieldaid_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("training/results/fieldaid_eval_results.json"))
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--adapter", action="store_true", help="Load --model-name as a PEFT adapter on top of --base-model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. Install them with: python -m pip install -r requirements-train.txt"
        ) from exc

    cases = load_jsonl(args.evals)
    model_ref = args.base_model if args.adapter else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.model_name)
    model.eval()

    predictions: list[str] = []
    for case in cases:
        prompt = (
            f"Scenario type: {case['scenario_type']}\n"
            f"Location: {case['location']}\n\n"
            f"Responder input:\n{case['input']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    score = score_predictions(cases, predictions)
    result = {
        "model_name": args.model_name,
        "base_model": args.base_model if args.adapter else None,
        "total": score.total,
        "valid_json": score.valid_json,
        "schema_ok": score.schema_ok,
        "urgency_ok": score.urgency_ok,
        "expected_terms_ok": score.expected_terms_ok,
        "forbidden_terms_ok": score.forbidden_terms_ok,
        "json_rate": score.json_rate,
        "safety_rate": score.safety_rate,
        "cases": [
            {
                "id": case["id"],
                "prediction": prediction,
            }
            for case, prediction in zip(cases, predictions)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
