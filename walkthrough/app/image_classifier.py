from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "fieldaid-medic-image-classifier-final" / "best.pt"


def classify_disaster_image(image_path: Path, top_k: int = 4, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    """Run the optional local MEDIC disaster classifier.

    This model is deliberately optional. FieldAid must still run on machines
    that have only Ollama/Gemma installed, so import and model-load failures are
    returned as structured unavailable states instead of startup errors.
    """
    if image_path is None or not image_path.exists():
        return _unavailable("image file is missing")
    if not checkpoint_path.exists():
        return _unavailable("classifier checkpoint is missing")
    try:
        model_bundle = _load_classifier(str(checkpoint_path))
        torch = model_bundle["torch"]
        image = model_bundle["image_open"](image_path).convert("RGB")
        tensor = model_bundle["transform"](image).unsqueeze(0)
        with torch.no_grad():
            logits = model_bundle["model"](tensor)
            probs = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probs, k=min(top_k, len(model_bundle["classes"])))
        predictions = [
            {
                "label": model_bundle["classes"][int(index)],
                "confidence": round(float(value), 6),
            }
            for value, index in zip(top.values, top.indices)
        ]
    except Exception as exc:  # pragma: no cover - depends on optional runtime packages
        return _unavailable(f"classifier failed: {exc}")

    top_prediction = predictions[0] if predictions else {"label": None, "confidence": 0.0}
    top_confidence = float(top_prediction["confidence"] or 0.0)
    return {
        "available": True,
        "model": model_bundle["model_name"],
        "checkpoint": checkpoint_path.name,
        "classes": model_bundle["classes"],
        "predictions": predictions,
        "top_label": top_prediction["label"],
        "top_confidence": top_confidence,
        "confidence_band": _confidence_band(top_confidence),
        "validation_accuracy": model_bundle.get("validation_accuracy"),
    }


@lru_cache(maxsize=2)
def _load_classifier(checkpoint_path: str) -> dict[str, Any]:
    import torch
    from PIL import Image
    from torch import nn
    from torchvision import models, transforms

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = [str(item) for item in checkpoint["classes"]]
    image_size = int(checkpoint.get("image_size", 224))
    model_name = str(checkpoint.get("model_name", "mobilenet_v3_small"))
    model = _build_model(model_name, len(classes), models, nn)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {
        "torch": torch,
        "image_open": Image.open,
        "model": model,
        "model_name": model_name,
        "classes": classes,
        "transform": transform,
        "validation_accuracy": checkpoint.get("val_acc"),
    }


def _build_model(name: str, num_classes: int, models: Any, nn: Any) -> Any:
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.70:
        return "likely"
    if confidence >= 0.45:
        return "possible"
    return "uncertain"


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "model": "fieldaid-medic-image-classifier",
        "checkpoint": None,
        "classes": [],
        "predictions": [],
        "top_label": None,
        "top_confidence": 0.0,
        "confidence_band": "unavailable",
        "reason": reason,
    }
