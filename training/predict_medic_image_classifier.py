from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FieldAid MEDIC image classifier on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/fieldaid-medic-image-classifier/best.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image
    from torch import nn
    from torchvision import models, transforms

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 224))
    model_name = checkpoint.get("model_name", "mobilenet_v3_small")
    model = build_model(model_name, len(classes), models, nn)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        logits = model(transform(image).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
    top = torch.topk(probs, k=min(args.top_k, len(classes)))
    result = [
        {"label": classes[int(index)], "confidence": float(value)}
        for value, index in zip(top.values, top.indices)
    ]
    print(json.dumps({"image": str(args.image), "predictions": result}, indent=2))


def build_model(name: str, num_classes: int, models, nn):
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


if __name__ == "__main__":
    main()
