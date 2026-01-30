from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from .model import SimpleCNN

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

class Predictor:
    def __init__(self, model_path: str, device: str | None = None, logger=None):
        self.logger = logger
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # CIFAR-10 expects 32x32. Uploaded images can be any size.
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        if self.logger:
            self.logger.info(f"Model loaded from {self.model_path} on device={self.device}")

    def predict(self, pil_image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        x = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_k = max(1, min(top_k, 10))
        values, indices = torch.topk(probs, k=top_k)

        topk: List[Dict[str, Any]] = []
        for p, idx in zip(values.tolist(), indices.tolist()):
            topk.append({"label": CIFAR10_CLASSES[idx], "probability": float(p)})

        return {"top_prediction": topk[0], "top_k": topk}
