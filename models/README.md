# FieldAid YOLO Models

Place custom disaster detection weights here:

```text
models/disaster_yolo.pt
```

The Video Scan backend prefers `models/disaster_yolo.pt` when present. If it is absent, FieldAid tries a lightweight pretrained Ultralytics YOLO model (`yolo11n.pt`). If YOLO is unavailable in the runtime, the API falls back to sampled frames and manual-review observations rather than claiming a route or structure is safe.
