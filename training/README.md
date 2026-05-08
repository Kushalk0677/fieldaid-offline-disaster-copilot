# FieldAid Gemma 4 Fine-Tuning

This folder contains a hackathon-ready LoRA fine-tuning path for adapting Gemma 4 to FieldAid disaster-response outputs.

The goal is not to train a foundation model from scratch. The intended submission artifact is a small, publishable LoRA adapter or merged/GGUF checkpoint that makes Gemma 4 better at:

- valid FieldAid JSON
- shelter and route triage
- fire/smoke/flood wording
- Discord/radio-ready handoff drafts
- refusal of unsafe bridge/building/medicine certainty
- explicit human verification flags

## Recommended Runtime

Use a Linux CUDA GPU environment such as Kaggle GPU, Colab, or WSL2 with NVIDIA CUDA. This Windows project venv currently has CPU-only PyTorch, so it can validate assets but should not be used for the actual train.

## Files

- `data/fieldaid_sft_seed.jsonl` - seed supervised fine-tuning examples.
- `data/fieldaid_vision_manifest.example.jsonl` - example multimodal image manifest rows.
- `evals/fieldaid_eval.jsonl` - small benchmark for JSON validity and safety behavior.
- `train_fieldaid_lora.py` - Unsloth LoRA training script.
- `train_fieldaid_vision_lora.py` - Unsloth multimodal vision LoRA training script.
- `build_vision_manifest.py` - creates image+JSON rows from labeled disaster image folders.
- `evaluate_fieldaid_adapter.py` - benchmark runner for a trained adapter or merged model.
- `validate_training_assets.py` - fast local JSONL/schema validation.
- `Modelfile.fieldaid-gemma4-e4b` - Ollama import template for a GGUF export.
- `SSH_GPU_RUNBOOK.md` - end-to-end remote GPU instructions.

## Validate Locally

```powershell
python -m training.validate_training_assets
```

## Train On GPU

```bash
python -m pip install -r requirements-train.txt
python -m training.train_fieldaid_lora \
  --model-name google/gemma-4-E4B-it \
  --data training/data/fieldaid_sft_seed.jsonl \
  --output-dir models/fieldaid-gemma4-e4b-lora \
  --epochs 3
```

## Train With Images

Create labeled image folders:

```text
training/images/fire
training/images/smoke
training/images/flood
training/images/road_damage
training/images/building_damage
training/images/unknown
```

Build a manifest:

```bash
python -m training.build_vision_manifest \
  --image-root training/images \
  --output training/data/fieldaid_vision_manifest.jsonl
```

Run a GPU smoke train:

```bash
python -m training.train_fieldaid_vision_lora \
  --model-name google/gemma-4-E4B-it \
  --data training/data/fieldaid_vision_manifest.jsonl \
  --output-dir models/fieldaid-gemma4-e4b-vision-lora-smoke \
  --max-records 8 \
  --epochs 0.05
```

Then run the full train:

```bash
python -m training.train_fieldaid_vision_lora \
  --model-name google/gemma-4-E4B-it \
  --data training/data/fieldaid_vision_manifest.jsonl \
  --output-dir models/fieldaid-gemma4-e4b-vision-lora \
  --epochs 2
```

See `training/SSH_GPU_RUNBOOK.md` for the SSH workflow.

Optional export commands:

```bash
python -m training.train_fieldaid_lora \
  --model-name google/gemma-4-E4B-it \
  --output-dir models/fieldaid-gemma4-e4b-lora \
  --epochs 3 \
  --merge-16bit \
  --export-gguf
```

## Benchmark

For an adapter:

```bash
python -m training.evaluate_fieldaid_adapter \
  --adapter \
  --base-model google/gemma-4-E4B-it \
  --model-name models/fieldaid-gemma4-e4b-lora \
  --output training/results/fieldaid_eval_results.json
```

The benchmark records:

- valid JSON rate
- required schema key rate
- urgency match rate
- expected safety term coverage
- forbidden unsafe phrase avoidance

## Ollama Import

After exporting a GGUF file, place it beside the Modelfile and run:

```bash
ollama create fieldaid-gemma4:e4b -f training/Modelfile.fieldaid-gemma4-e4b
```

Then choose `fieldaid-gemma4:e4b` in FieldAid's model dropdown.

## Submission Notes

If we train this adapter for the Kaggle submission, publish:

- LoRA adapter or merged/GGUF weights
- training data source description
- eval JSONL
- benchmark output JSON
- model card describing safety boundaries and limitations
