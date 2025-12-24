from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from mrag.storage.manifest import ImageAsset


@dataclass(frozen=True)
class ImageHit:
    image_id: str
    doc_id: str
    page_index: int
    score: float
    path: str
    kind: str = "embedded"
    ocr_text: str | None = None


def build_image_index(
    *,
    out_dir: Path,
    images: list[ImageAsset],
    embeddings: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(images) != embeddings.shape[0]:
        raise ValueError("images and embeddings size mismatch")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(out_dir / "image.faiss"))
    with (out_dir / "image_assets.jsonl").open("w", encoding="utf-8") as f:
        for im in images:
            f.write(json.dumps(asdict(im), ensure_ascii=False) + "\n")


def load_image_index(index_dir: Path) -> tuple[faiss.Index, list[ImageAsset]]:
    index = faiss.read_index(str(index_dir / "image.faiss"))
    images: list[ImageAsset] = []
    with (index_dir / "image_assets.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            images.append(ImageAsset(**d))
    return index, images


def search_image(
    *,
    index: faiss.Index,
    images: list[ImageAsset],
    query_vec: np.ndarray,
    top_k: int,
) -> list[ImageHit]:
    if query_vec.ndim == 1:
        q = query_vec.reshape(1, -1)
    else:
        q = query_vec
    scores, idxs = index.search(q.astype(np.float32), top_k)
    hits: list[ImageHit] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist(), strict=False):
        if idx < 0 or idx >= len(images):
            continue
        im = images[idx]
        hits.append(
            ImageHit(
                image_id=im.image_id,
                doc_id=im.doc_id,
                page_index=im.page_index,
                score=float(score),
                path=im.path,
                kind=getattr(im, "kind", "embedded"),
                ocr_text=im.ocr_text,
            )
        )
    return hits


