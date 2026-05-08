from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path


def disable_torch_compile_for_legacy_gpu() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


SYSTEM_PROMPT = """You are FieldAid, an offline disaster-response copilot.
Return only valid JSON for trained responders.
Never claim to replace emergency services, clinicians, engineers, or incident command.
Never declare a road, bridge, building, fire scene, or medical situation safe.
Medical, structural, and life-safety outputs must include human verification flags."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 for FieldAid with Unsloth LoRA.")
    parser.add_argument("--model-name", default="google/gemma-4-E4B-it", help="Base Gemma 4 instruction model on Hugging Face.")
    parser.add_argument("--data", type=Path, default=Path("training/data/fieldaid_sft_seed.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fieldaid-gemma4-e4b-lora"))
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--merge-16bit", action="store_true", help="Also write merged 16-bit weights.")
    parser.add_argument("--export-gguf", action="store_true", help="Also export a GGUF model if supported by the installed Unsloth build.")
    return parser.parse_args()


def main() -> None:
    disable_torch_compile_for_legacy_gpu()
    args = parse_args()
    try:
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them on a CUDA GPU environment with: "
            "python -m pip install -r requirements-train.txt"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files=str(args.data), split="train")

    def format_record(record: dict) -> dict:
        output = json.dumps(record["output"], ensure_ascii=False)
        user_content = (
            f"Scenario type: {record['scenario_type']}\n"
            f"Location: {record['location']}\n\n"
            f"Instruction: {record['instruction']}\n\n"
            f"Responder input:\n{record['input']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_record, remove_columns=dataset.column_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "optim": "adamw_8bit",
        "fp16": True,
        "bf16": False,
        "seed": args.seed,
        "report_to": "none",
    }
    config_signature = inspect.signature(SFTConfig).parameters
    if "dataset_text_field" in config_signature:
        config_kwargs["dataset_text_field"] = "text"
    if "warmup_ratio" in config_signature:
        config_kwargs["warmup_ratio"] = 0.05
    elif "warmup_steps" in config_signature:
        config_kwargs["warmup_steps"] = 0

    trainer_kwargs = {
        "model": model,
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

    if args.merge_16bit:
        merged_dir = args.output_dir.with_name(args.output_dir.name + "-merged-16bit")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    if args.export_gguf:
        gguf_dir = args.output_dir.with_name(args.output_dir.name + "-gguf")
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method="q4_k_m")


if __name__ == "__main__":
    main()
