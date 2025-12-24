from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class PageText:
    doc_id: str
    page_index: int
    text: str


@dataclass
class TextBlock:
    # PDF page coordinates (points): [x0, y0, x1, y1]
    bbox: list[float]
    text: str


@dataclass
class PageBlocks:
    doc_id: str
    page_index: int
    page_width: float
    page_height: float
    blocks: list[TextBlock]


@dataclass
class ImageAsset:
    doc_id: str
    page_index: int
    image_id: str
    path: str
    # "page" for rendered full-page images; "embedded" for images extracted from PDF objects
    kind: str = "embedded"
    # bbox in PDF page coordinates if available; otherwise null
    bbox: list[float] | None = None
    ocr_text: str | None = None


@dataclass
class DocumentRecord:
    doc_id: str
    source_path: str
    num_pages: int


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_records(
    base_dir: Path,
    docs: list[DocumentRecord],
    page_texts: list[PageText],
    images: list[ImageAsset],
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(base_dir / "docs.jsonl", (asdict(d) for d in docs))
    write_jsonl(base_dir / "page_texts.jsonl", (asdict(t) for t in page_texts))
    write_jsonl(base_dir / "images.jsonl", (asdict(i) for i in images))


def load_records(base_dir: Path) -> tuple[list[DocumentRecord], list[PageText], list[ImageAsset]]:
    docs_raw = read_jsonl(base_dir / "docs.jsonl")
    pages_raw = read_jsonl(base_dir / "page_texts.jsonl")
    images_raw = read_jsonl(base_dir / "images.jsonl")
    docs = [DocumentRecord(**d) for d in docs_raw]
    pages = [PageText(**p) for p in pages_raw]
    images = [ImageAsset(**i) for i in images_raw]
    return docs, pages, images


def dump_page_blocks(base_dir: Path, page_blocks: list[PageBlocks]) -> None:
    if not page_blocks:
        return
    write_jsonl(base_dir / "page_blocks.jsonl", (asdict(p) for p in page_blocks))


def load_page_blocks(base_dir: Path) -> list[PageBlocks]:
    path = base_dir / "page_blocks.jsonl"
    if not path.exists():
        return []
    raw = read_jsonl(path)
    out: list[PageBlocks] = []
    for r in raw:
        blocks = [TextBlock(**b) for b in r.get("blocks", [])]
        out.append(
            PageBlocks(
                doc_id=r["doc_id"],
                page_index=int(r["page_index"]),
                page_width=float(r["page_width"]),
                page_height=float(r["page_height"]),
                blocks=blocks,
            )
        )
    return out


