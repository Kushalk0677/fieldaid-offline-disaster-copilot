# MEDIC by QCRI for FieldAid

This folder is separate from the FieldAid app runtime. It downloads and prepares the public `QCRI/MEDIC` disaster-image dataset from Hugging Face for Gemma 4 multimodal fine-tuning.

Dataset: `QCRI/MEDIC`

License shown on Hugging Face: CC BY-NC-SA 4.0. Use it for research/hackathon work; review licensing before commercial use.

## Folder Layout

After download:

```text
modeltraining/medic_qcri/
  raw/
    metadata_train.jsonl
    metadata_validation.jsonl
    metadata_test.jsonl
  images/
    train/
    validation/
    test/
  fieldaid/
    images/
      fire/
      flood/
      road_damage/
      building_damage/
      affected_people/
      unknown/
    fieldaid_vision_manifest.jsonl
    label_summary.json
```

## Install

Use a training venv, not the app venv if possible:

```powershell
python -m venv .venv-medic
.\.venv-medic\Scripts\Activate.ps1
python -m pip install -r modeltraining\medic_qcri\requirements.txt
```

## Download Full Dataset

```powershell
python modeltraining\medic_qcri\download_medic.py
```

## Smoke Download

Use this first if you want to test access and disk behavior:

```powershell
python modeltraining\medic_qcri\download_medic.py --max-rows 25
```

If a previous conversion crashed, keep the Hugging Face cache but rebuild outputs cleanly:

```powershell
python modeltraining\medic_qcri\download_medic.py --clean-output --overwrite
```

To convert only a quarter of the dataset:

```powershell
python modeltraining\medic_qcri\download_medic.py --clean-output --overwrite --fraction 0.25
```

On Linux, reduce duplicate image storage by hardlinking FieldAid label folders back to the split images:

```bash
python medic_qcri/download_medic.py --clean-output --overwrite --fraction 0.25 --link-fieldaid-images
```

If the Codex local sandbox blocks outbound HTTP, run the same command with network approval or run it on your SSH GPU machine:

```bash
cd ~/fieldaid
source .venv-train/bin/activate
python -m pip install -r modeltraining/medic_qcri/requirements.txt
python modeltraining/medic_qcri/download_medic.py --max-rows 25
python modeltraining/medic_qcri/download_medic.py
```

The downloader uses `modeltraining/medic_qcri/hf_cache` so interrupted downloads can resume through Hugging Face's dataset cache.

## Use With FieldAid Vision Trainer

Copy or point the generated manifest into the training folder:

```powershell
Copy-Item modeltraining\medic_qcri\fieldaid\fieldaid_vision_manifest.jsonl training\data\fieldaid_vision_manifest.jsonl
```

Then train on the GPU box using `training/SSH_GPU_RUNBOOK.md`.
