from pathlib import Path

from app.image_classifier import classify_disaster_image


def test_classifier_reports_unavailable_for_missing_checkpoint(tmp_path: Path):
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"not actually an image")
    result = classify_disaster_image(image, checkpoint_path=tmp_path / "missing.pt")
    assert result["available"] is False
    assert result["confidence_band"] == "unavailable"
    assert "checkpoint" in result["reason"]
