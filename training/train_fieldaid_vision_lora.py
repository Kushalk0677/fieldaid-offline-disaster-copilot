from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

from training.train_fieldaid_lora import SYSTEM_PROMPT


def disable_torch_compile_for_legacy_gpu() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a multimodal Gemma 4 FieldAid adapter with Unsloth.")
    parser.add_argument("--model-name", default="google/gemma-4-E4B-it")
    parser.add_argument("--data", type=Path, default=Path("training/data/fieldaid_vision_manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fieldaid-gemma4-e4b-vision-lora"))
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--finetune-vision-layers", action="store_true", help="Train vision tower LoRA layers. Leave off for 12 GB/P100 runs.")
    parser.add_argument("--freeze-language-layers", action="store_true", help="Do not train language layers.")
    parser.add_argument("--freeze-attention-modules", action="store_true", help="Do not train attention modules.")
    parser.add_argument("--freeze-mlp-modules", action="store_true", help="Do not train MLP modules.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    disable_torch_compile_for_legacy_gpu()
    args = parse_args()
    try:
        from PIL import Image
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as exc:
        raise SystemExit(
            "Vision training dependencies are missing. Run on CUDA Linux/WSL/Kaggle with: "
            "python -m pip install -r requirements-train.txt pillow"
        ) from exc

    records = load_records(args.data, args.max_records)
    if not records:
        raise SystemExit(f"No vision training records found in {args.data}")

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=not args.freeze_language_layers,
        finetune_attention_modules=not args.freeze_attention_modules,
        finetune_mlp_modules=not args.freeze_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
    )
    FastVisionModel.for_training(model)

    def to_conversation(record: dict) -> dict:
        image_path = Path(record["image"])
        image = Image.open(image_path).convert("RGB")
        user_text = (
            f"Scenario type: {record['scenario_type']}\n"
            f"Location: {record['location']}\n\n"
            f"Instruction: {record['instruction']}"
        )
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(record["output"], ensure_ascii=False)}]},
            ]
        }

    dataset = Dataset.from_list([to_conversation(record) for record in records])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "optim": "adamw_8bit",
        "fp16": True,
        "bf16": False,
        "remove_unused_columns": False,
        "seed": args.seed,
        "report_to": "none",
    }
    config_signature = inspect.signature(SFTConfig).parameters
    if "warmup_ratio" in config_signature:
        config_kwargs["warmup_ratio"] = 0.03
    elif "warmup_steps" in config_signature:
        config_kwargs["warmup_steps"] = 0
    if "dataset_text_field" in config_signature:
        config_kwargs["dataset_text_field"] = ""
    if "dataset_kwargs" in config_signature:
        config_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    trainer_kwargs = {
        "model": model,
        "data_collator": UnslothVisionDataCollator(model, tokenizer),
        "train_dataset": dataset,
        "args": SFTConfig(**config_kwargs),
    }
    if "processing_class" in inspect.signature(SFTTrainer).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def load_records(path: Path, max_records: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if Path(record["image"]).exists():
                records.append(record)
            if max_records and len(records) >= max_records:
                break
    return records


if __name__ == "__main__":
    main()
