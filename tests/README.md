# `tests/`

Automated verification for FieldAid.

- `test_api.py`: FastAPI endpoint behavior for analysis, grounding, outbox, sync, and video scan.
- `test_core.py`: incident prioritization and core response logic.
- `test_video_scan.py`: frame sampling, YOLO model selection, classifier aggregation, and video safety behavior.
- `test_image_classifier.py`: optional classifier failure behavior.
- `test_training_assets.py`: dataset downloader and training asset validation.

Run from the repository root:

```powershell
python -m pytest -q tests
```

On locked-down Windows temp directories, use a writable temp base:

```powershell
$env:TEMP='C:\tmp'; $env:TMP='C:\tmp'; python -m pytest -q tests --basetemp C:\tmp\fieldaid_pytest -p no:cacheprovider
```
