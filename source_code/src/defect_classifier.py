"""
src/defect_classifier.py
========================
Lightweight defect classifier for the Conveyor Belt CV System.

Model:  MobileNetV3-Small (torchvision) fine-tuned for binary classification:
    Class 0 → Normal
    Class 1 → Defective

In ``simulate=True`` mode (default when no weights are available) the
classifier returns a random but plausible result — useful for demos and
integration testing without real training data.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    label: str           # "Normal" or "Defective"
    confidence: float    # Probability of the predicted class (0–1)
    class_id: int        # 0 = Normal, 1 = Defective
    latency_ms: float = 0.0

    @property
    def is_defective(self) -> bool:
        return self.class_id == 1


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class DefectClassifier:
    """
    MobileNetV3-Small based binary defect classifier.

    Args:
        weights_path:      Path to fine-tuned .pt weights.  If None or file
                           does not exist, falls back to simulate mode.
        input_size:        Crop resize resolution (square).
        defect_threshold:  Probability above which a product is "Defective".
        simulate:          Force simulation mode regardless of weights.
        num_classes:       Number of output classes (default 2).
    """

    LABELS = {0: "Normal", 1: "Defective"}

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        input_size: int = 224,
        defect_threshold: float = 0.55,
        simulate: bool = True,
        num_classes: int = 2,
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self.input_size = input_size
        self.defect_threshold = defect_threshold
        self.num_classes = num_classes
        self.latency_ms: float = 0.0

        self._model = None
        self._transform = None
        self._device = None
        self._simulate = simulate

        if not simulate:
            self._try_load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self):
        """Try to load real model weights; fall back to simulation on failure."""
        if self.weights_path is None or not self.weights_path.exists():
            print(
                f"[DefectClassifier] Weights not found at "
                f"'{self.weights_path}'. Using simulation mode."
            )
            self._simulate = True
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = models.mobilenet_v3_small(
                weights=None,
                num_classes=self.num_classes,
            )
            state = torch.load(str(self.weights_path), map_location=self._device)
            model.load_state_dict(state)
            model.eval()
            model.to(self._device)
            self._model = model

            self._transform = transforms.Compose(
                [
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self._simulate = False
            print(f"[DefectClassifier] Loaded weights from '{self.weights_path}'.")

        except Exception as exc:
            print(f"[DefectClassifier] Failed to load model ({exc}). Using simulation.")
            self._simulate = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classify a single product crop image.

        Args:
            crop: H×W×3 uint8 BGR numpy array (from cv2).

        Returns:
            ``ClassificationResult`` with label and confidence.
        """
        t0 = time.perf_counter()

        if self._simulate:
            result = self._simulate_classify(crop)
        else:
            result = self._real_classify(crop)

        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def classify_batch(self, crops: List[np.ndarray]) -> List[ClassificationResult]:
        """Classify a batch of crops."""
        return [self.classify(c) for c in crops]

    def _simulate_classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Simulated classifier — 85% probability of Normal, 15% Defective.
        Adds a tiny sleep to mimic real inference latency.
        """
        time.sleep(random.uniform(0.001, 0.004))  # 1–4 ms
        defect_prob = random.uniform(0.0, 1.0)

        # Use crop brightness as a naive heuristic when simulating
        # (makes the demo slightly more visually consistent)
        if crop is not None and crop.size > 0:
            brightness = float(np.mean(crop))
            # Very bright or very dark patches are slightly more "defective"
            defect_prob = 0.15 + 0.10 * max(0.0, (abs(brightness - 128.0) / 128.0 - 0.5))

        if defect_prob >= self.defect_threshold:
            return ClassificationResult(label="Defective", confidence=defect_prob, class_id=1)
        else:
            return ClassificationResult(label="Normal", confidence=1.0 - defect_prob, class_id=0)

    def _real_classify(self, crop: np.ndarray) -> ClassificationResult:
        """Run MobileNetV3 inference on the crop."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import cv2

        # BGR → RGB PIL
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        cls_id = int(np.argmax(probs))
        label = self.LABELS.get(cls_id, "Unknown")
        conf = float(probs[cls_id])

        return ClassificationResult(label=label, confidence=conf, class_id=cls_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_simulation(self) -> bool:
        return self._simulate

    def __repr__(self) -> str:
        mode = "simulate" if self._simulate else "real"
        return (
            f"DefectClassifier(mode={mode!r}, "
            f"threshold={self.defect_threshold}, "
            f"input_size={self.input_size})"
        )
