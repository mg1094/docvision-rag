from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError("PyTorch not available. Install torch for reranker.") from e


@dataclass
class TextReranker:
    model_name: str
    device: str = "auto"
    batch_size: int = 32

    def __post_init__(self) -> None:
        _require_torch()
        import torch
        from sentence_transformers import CrossEncoder

        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        # CrossEncoder 会下载/加载一个序列分类模型，用于(query, passage)打分
        self._model = CrossEncoder(self.model_name, device=self._device, max_length=1024)

    def score(self, query: str, passages: Iterable[str]) -> np.ndarray:
        ps = list(passages)
        if not ps:
            return np.zeros((0,), dtype=np.float32)
        pairs = [(query, p) for p in ps]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


