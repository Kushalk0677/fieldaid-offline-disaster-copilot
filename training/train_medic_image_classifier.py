from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight disaster image classifier from MEDIC label folders.")
    parser.add_argument("--image-root", type=Path, default=Path("medic_qcri/fieldaid/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fieldaid-medic-image-classifier"))
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=0, help="Optional cap per class for smoke tests.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise SystemExit("Missing image classifier dependencies. Install torch, torchvision, and pillow.") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, classes = collect_samples(args.image_root, args.max_per_class, args.seed)
    if len(classes) < 2:
        raise SystemExit(f"Need at least 2 class folders under {args.image_root}")
    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = ImagePathDataset(train_samples, train_transform, Image)
    val_ds = ImagePathDataset(val_samples, val_transform, Image)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(args.model, len(classes), models, nn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row))
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": args.image_size,
                    "val_acc": val_acc,
                },
                args.output_dir / "best.pt",
            )

    (args.output_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "classes": classes,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "best_val_acc": best_acc,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def collect_samples(image_root: Path, max_per_class: int, seed: int) -> tuple[list[tuple[Path, int]], list[str]]:
    classes = sorted([path.name for path in image_root.iterdir() if path.is_dir()])
    samples: list[tuple[Path, int]] = []
    rng = random.Random(seed)
    for label_id, class_name in enumerate(classes):
        paths = sorted(path for path in (image_root / class_name).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(paths)
        if max_per_class:
            paths = paths[:max_per_class]
        samples.extend((path, label_id) for path in paths)
    rng.shuffle(samples)
    return samples, classes


def split_samples(samples: list[tuple[Path, int]], val_fraction: float, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)
    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        val_count = max(1, int(len(label_samples) * val_fraction)) if len(label_samples) > 1 else 0
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class ImagePathDataset:
    def __init__(self, samples, transform, image_module):
        self.samples = samples
        self.transform = transform
        self.image_module = image_module

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            image = self.image_module.open(path).convert("RGB")
        except Exception:
            image = self.image_module.new("RGB", (224, 224), color=(0, 0, 0))
        return self.transform(image), label


def build_model(name: str, num_classes: int, models, nn):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool) -> tuple[float, float]:
    import torch

    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


if __name__ == "__main__":
    main()
