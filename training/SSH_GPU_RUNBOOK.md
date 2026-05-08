# FieldAid Vision Training Over SSH

Use this when you have a CUDA GPU machine reachable over SSH. The local Windows venv can validate data, but real Gemma 4 vision LoRA training should run on a Linux CUDA environment.

## 1. Prepare Image Folders

Create a labeled image folder locally or on the GPU box:

```text
training/images/
  fire/
  smoke/
  flood/
  road_damage/
  building_damage/
  unknown/
```

Keep labels simple. The first training pass should teach FieldAid operational language, not perfect computer vision. Aim for:

- 200-500 images per class for a first useful adapter
- 50-100 images per class for a quick smoke train
- separate held-out eval images that are not in train

Useful public dataset families to adapt:

- crisis/disaster social media images
- flood and fire scene images
- road/bridge/building damage images
- satellite damage datasets only if the app will handle satellite-like images

For each image, the trainer expects a FieldAid JSON target, not just a class label. `build_vision_manifest.py` creates safe starter targets from class folders.

## 2. Build The Vision Manifest

From the project root:

```powershell
python -m training.build_vision_manifest `
  --image-root training/images `
  --output training/data/fieldaid_vision_manifest.jsonl
```

Manually review a sample:

```powershell
Get-Content training\data\fieldaid_vision_manifest.jsonl -TotalCount 3
```

## 3. Copy Project To GPU Machine

From PowerShell, replace the SSH target:

```powershell
scp -r . user@gpu-host:~/fieldaid
```

For large image folders, prefer `rsync` from WSL/Git Bash:

```bash
rsync -av --progress ./ user@gpu-host:~/fieldaid/
```

Skip `.venv`, `data/uploads`, old pytest caches, and server logs if transferring manually.

## 4. Set Up GPU Environment

On the GPU host:

```bash
cd ~/fieldaid
nvidia-smi
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-train.txt pillow
```

If the Gemma repo requires Hugging Face auth:

```bash
huggingface-cli login
```

## 5. Smoke-Test With A Few Images

```bash
python -m training.train_fieldaid_vision_lora \
  --model-name google/gemma-4-E4B-it \
  --data training/data/fieldaid_vision_manifest.jsonl \
  --output-dir models/fieldaid-gemma4-e4b-vision-lora-smoke \
  --max-records 8 \
  --epochs 0.05
```

This should prove dependencies, model access, CUDA memory, and dataset image paths.

## 6. Train The Vision Adapter

Start conservative:

```bash
python -m training.train_fieldaid_vision_lora \
  --model-name google/gemma-4-E4B-it \
  --data training/data/fieldaid_vision_manifest.jsonl \
  --output-dir models/fieldaid-gemma4-e4b-vision-lora \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4
```

If VRAM is tight:

- keep `batch-size 1`
- lower `max-seq-length` to `2048`
- use fewer records first
- use text-only LoRA as fallback

If quality is weak:

- add more image diversity
- write more specific JSON targets for hard cases
- add negative/unknown examples
- add separate before/after route damage examples
- run 3-4 epochs only after the dataset is clean

## 7. Evaluate

The current eval runner is text-only and checks FieldAid JSON/safety behavior. Use it for the text adapter or merged model:

```bash
python -m training.evaluate_fieldaid_adapter \
  --adapter \
  --base-model google/gemma-4-E4B-it \
  --model-name models/fieldaid-gemma4-e4b-lora \
  --output training/results/fieldaid_eval_results.json
```

For the vision adapter, manually create a held-out `training/data/fieldaid_vision_eval.jsonl` and inspect predictions on images. The important metrics for the writeup:

- disaster type accuracy
- valid JSON rate
- human-verification flag rate
- unsafe-safe-claim rate, which should be zero
- Discord/SMS usefulness

## 8. Bring Weights Back

From local PowerShell:

```powershell
scp -r user@gpu-host:~/fieldaid/models/fieldaid-gemma4-e4b-vision-lora .\models\
scp -r user@gpu-host:~/fieldaid/training/results .\training\
```

Publish the LoRA adapter and benchmark results for the hackathon if used in the submission.

## 9. Ollama Path

Ollama usually wants a GGUF or supported import format. If Unsloth export works for the selected Gemma 4 build, export GGUF on the GPU box, copy it locally, then:

```bash
ollama create fieldaid-gemma4:e4b -f training/Modelfile.fieldaid-gemma4-e4b
```

Then use `fieldaid-gemma4:e4b` in the FieldAid UI.

## Data Quality Rules

- Do not train on copyrighted/private images unless the license allows it.
- Keep disaster categories balanced.
- Include unknown/ambiguous images so the model learns to say "manual review required."
- Do not label a bridge/building/road as safe.
- Keep medical examples focused on escalation, not medication advice.
- Keep a held-out eval set from different sources than training where possible.
