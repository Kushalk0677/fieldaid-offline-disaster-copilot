# `models/fieldaid-medic-image-classifier-final/`

Final lightweight disaster-image classifier used by FieldAid.

Files:

- `best.pt`: MobileNetV3 Small checkpoint trained on a QCRI/MEDIC disaster-image subset.
- `classes.json`: class labels used by the checkpoint.
- `metrics.json`: training and validation metrics.

Classes:

- `building_damage`
- `fire`
- `flood`
- `road_damage`

The app uses this model in `app/image_classifier.py` for Trust photo uploads and video frame aggregation. It is supporting evidence only; FieldAid never declares routes, bridges, or structures safe from imagery.
