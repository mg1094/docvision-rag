from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from mrag.ingest.chunking import TextChunk


@dataclass(frozen=True)
class TextHit:
    chunk_id: str
    doc_id: str
    page_index: int
    page_end: int | None = None
    bboxes: list[list[float]] | None = None
    bboxes_end: list[list[float]] | None = None
    score: float
    text: str


def build_text_index(
    *,
    out_dir: Path,
    chunks: list[TextChunk],
    embeddings: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("chunks and embeddings size mismatch")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(out_dir / "text.faiss"))
    with (out_dir / "text_chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_text_index(index_dir: Path) -> tuple[faiss.Index, list[TextChunk]]:
    index = faiss.read_index(str(index_dir / "text.faiss"))
    chunks: list[TextChunk] = []
    with (index_dir / "text_chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(TextChunk(**d))
    return index, chunks


def search_text(
    *,
    index: faiss.Index,
    chunks: list[TextChunk],
    query_vec: np.ndarray,
    top_k: int,
) -> list[TextHit]:
    if query_vec.ndim == 1:
        q = query_vec.reshape(1, -1)
    else:
        q = query_vec
    scores, idxs = index.search(q.astype(np.float32), top_k)
    hits: list[TextHit] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist(), strict=False):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        hits.append(
            TextHit(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                page_index=c.page_index,
                page_end=getattr(c, "page_end", None),
                bboxes=getattr(c, "bboxes", None),
                bboxes_end=getattr(c, "bboxes_end", None),
                score=float(score),
                text=c.text,
            )
        )
    return hits


