from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch not available. Install torch for embedding/model inference."
        ) from e


@dataclass
class TextEmbedder:
    model_name: str
    device: str = "auto"
    batch_size: int = 32

    def __post_init__(self) -> None:
        _require_torch()
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        import torch

        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, 1), dtype=np.float32)

        embs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts_list), self.batch_size):
                batch = texts_list[i : i + self.batch_size]
                tok = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                tok = {k: v.to(self._device) for k, v in tok.items()}
                out = self._model(**tok)
                # mean pooling with attention mask
                last = out.last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).to(last.dtype)
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embs)


