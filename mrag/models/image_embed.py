from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch not available. Install torch for embedding/model inference."
        ) from e


@dataclass
class ImageEmbedder:
    model_name: str
    device: str = "auto"
    batch_size: int = 32

    def __post_init__(self) -> None:
        _require_torch()
        import torch
        from transformers import AutoModel, AutoProcessor

        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        import torch

        paths = [Path(p) for p in image_paths]
        if not paths:
            return np.zeros((0, 1), dtype=np.float32)

        embs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                batch_paths = paths[i : i + self.batch_size]
                imgs = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self._processor(images=imgs, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                # works for CLIP/SigLIP-like models with get_image_features()
                if hasattr(self._model, "get_image_features"):
                    feats = self._model.get_image_features(**inputs)
                else:
                    out = self._model(**inputs)
                    feats = out.last_hidden_state.mean(dim=1)
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                embs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embs)

    def encode_text(self, texts: Iterable[str]) -> np.ndarray:
        """
        用于“以文搜图”：把 query 文本编码到同一 embedding 空间。
        注意：CLIP/SigLIP 支持文本侧；如果模型不支持，会 fallback 到平均池化。
        """
        import torch

        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, 1), dtype=np.float32)

        embs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts_list), self.batch_size):
                batch = texts_list[i : i + self.batch_size]
                inputs = self._processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                if hasattr(self._model, "get_text_features"):
                    feats = self._model.get_text_features(**inputs)
                else:
                    out = self._model(**inputs)
                    feats = out.last_hidden_state.mean(dim=1)
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                embs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embs)


