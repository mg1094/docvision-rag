from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mrag.index.faiss_image import ImageHit
from mrag.index.faiss_text import TextHit


EvidenceType = Literal["text", "image"]


@dataclass(frozen=True)
class Evidence:
    type: EvidenceType
    doc_id: str
    page_index: int
    # for cross-page text evidence
    page_end: int | None = None
    score: float
    # text evidence
    chunk_id: str | None = None
    text: str | None = None
    text_bboxes: list[list[float]] | None = None
    text_bboxes_end: list[list[float]] | None = None
    # image evidence
    image_id: str | None = None
    image_path: str | None = None
    image_kind: str | None = None
    image_ocr_text: str | None = None


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_hits(
    *,
    text_hits: list[TextHit],
    image_hits: list[ImageHit],
    alpha: float,
    fuse_k: int,
) -> list[Evidence]:
    """
    简单融合策略：
    - 文本/图片各自做 min-max 归一化
    - 最终得分 = alpha * text_norm 或 (1-alpha) * image_norm
    - 合并后取 TopK
    """
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    t_norm = _minmax([h.score for h in text_hits])
    i_norm = _minmax([h.score for h in image_hits])

    evs: list[Evidence] = []
    for h, s in zip(text_hits, t_norm, strict=False):
        evs.append(
            Evidence(
                type="text",
                doc_id=h.doc_id,
                page_index=h.page_index,
                page_end=getattr(h, "page_end", None),
                score=alpha * float(s),
                chunk_id=h.chunk_id,
                text=h.text,
                text_bboxes=getattr(h, "bboxes", None),
                text_bboxes_end=getattr(h, "bboxes_end", None),
            )
        )
    for h, s in zip(image_hits, i_norm, strict=False):
        evs.append(
            Evidence(
                type="image",
                doc_id=h.doc_id,
                page_index=h.page_index,
                score=(1.0 - alpha) * float(s),
                image_id=h.image_id,
                image_path=h.path,
                image_kind=getattr(h, "kind", None),
                image_ocr_text=h.ocr_text,
            )
        )

    evs.sort(key=lambda e: e.score, reverse=True)
    return evs[: max(1, int(fuse_k))]


