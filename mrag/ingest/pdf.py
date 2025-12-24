from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import fitz  # PyMuPDF
from PIL import Image
from rich.console import Console

from mrag.storage.manifest import DocumentRecord, ImageAsset, PageBlocks, PageText, TextBlock


console = Console()


@dataclass(frozen=True)
class IngestResult:
    docs: list[DocumentRecord]
    page_texts: list[PageText]
    images: list[ImageAsset]
    page_blocks: list[PageBlocks]


def _stable_doc_id(p: Path) -> str:
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:12]
    return f"doc_{h}"


def _iter_pdfs(input_dir: Path, pdf_glob: str) -> Iterable[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    yield from sorted(input_dir.glob(pdf_glob))


def ingest_pdfs(
    *,
    input_dir: Path,
    pdf_glob: str,
    dataset_dir: Path,
    image_max_per_page: int = 8,
    render_pages_enabled: bool = False,
    render_pages_dpi: int = 200,
    ocr_enabled: bool = False,
    ocr_language: str = "ch",
    ocr_min_text_chars: int = 80,
    ocr_page_render_dpi: int = 200,
) -> IngestResult:
    images_dir = dataset_dir / "assets" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = dataset_dir / "assets" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    docs: list[DocumentRecord] = []
    page_texts: list[PageText] = []
    images: list[ImageAsset] = []
    page_blocks: list[PageBlocks] = []

    pdfs = list(_iter_pdfs(input_dir, pdf_glob))
    if not pdfs:
        console.print(f"[yellow]No PDFs matched {pdf_glob} under {input_dir}[/yellow]")
        return IngestResult(docs=docs, page_texts=page_texts, images=images, page_blocks=page_blocks)

    for pdf_path in pdfs:
        doc_id = _stable_doc_id(pdf_path)
        console.print(f"[cyan]Ingest[/cyan] {pdf_path} -> {doc_id}")

        pdf = fitz.open(pdf_path)
        ocr = _maybe_create_ocr(ocr_enabled=ocr_enabled, language=ocr_language)
        docs.append(
            DocumentRecord(
                doc_id=doc_id,
                source_path=str(pdf_path),
                num_pages=pdf.page_count,
            )
        )

        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            text = page.get_text("text") or ""

            # 版面信息（用于 B2 高亮）：提取文本块 bbox + 文本
            try:
                pw = float(page.rect.width)
                ph = float(page.rect.height)
                blocks_raw = page.get_text("blocks") or []
                blocks: list[TextBlock] = []
                for b in blocks_raw:
                    # (x0, y0, x1, y1, text, block_no, block_type, ...)
                    if not b or len(b) < 5:
                        continue
                    x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    bt = str(b[4]).strip()
                    # 过滤太短的噪声块
                    if len(bt) < 2:
                        continue
                    blocks.append(TextBlock(bbox=[x0, y0, x1, y1], text=bt))
                page_blocks.append(
                    PageBlocks(
                        doc_id=doc_id,
                        page_index=page_index,
                        page_width=pw,
                        page_height=ph,
                        blocks=blocks,
                    )
                )
            except Exception:
                # 不影响主流程：无 bbox 就退化为轻量版可视化
                pass

            # 企业文档常见痛点：矢量表格/图表 get_images() 抓不到
            # 解决：可选“整页渲染入图像索引”，同时 OCR 也复用该渲染页
            need_ocr = ocr is not None and len(text.strip()) < max(0, ocr_min_text_chars)
            need_page_render = bool(render_pages_enabled) or bool(need_ocr)
            if need_page_render:
                page_image_id = f"{doc_id}:p{page_index}:page"
                page_img_path = pages_dir / f"{page_image_id}.jpg"
                dpi = render_pages_dpi if render_pages_enabled else ocr_page_render_dpi
                _render_page_to_jpg(page=page, out_path=page_img_path, dpi=dpi)
                page_asset = ImageAsset(
                    doc_id=doc_id,
                    page_index=page_index,
                    image_id=page_image_id,
                    path=str(page_img_path),
                    kind="page",
                    bbox=None,
                    ocr_text=None,
                )
                if need_ocr:
                    ocr_text = _run_ocr(ocr, page_img_path)
                    if ocr_text:
                        page_asset.ocr_text = ocr_text
                        text = (text.strip() + "\n" + ocr_text.strip()).strip()
                images.append(page_asset)
            page_texts.append(PageText(doc_id=doc_id, page_index=page_index, text=text))

            # Extract images from page (raster only). For charts as vectors, a later step can render page crops.
            image_list = page.get_images(full=True)
            for img_i, img in enumerate(image_list[: max(0, image_max_per_page)]):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png")

                image_id = f"{doc_id}:p{page_index}:img{img_i}:{uuid.uuid4().hex[:8]}"
                out_path = images_dir / f"{image_id}.{ext}"
                out_path.write_bytes(img_bytes)

                # Normalize to RGB for later embedding
                try:
                    im = Image.open(out_path).convert("RGB")
                    im.save(out_path.with_suffix(".jpg"), quality=92)
                    out_path.unlink(missing_ok=True)
                    final_path = out_path.with_suffix(".jpg")
                except Exception:
                    final_path = out_path

                images.append(
                    ImageAsset(
                        doc_id=doc_id,
                        page_index=page_index,
                        image_id=image_id,
                        path=str(final_path),
                        kind="embedded",
                        bbox=None,
                        ocr_text=_run_ocr(ocr, Path(final_path)) if ocr is not None else None,
                    )
                )

        pdf.close()

    return IngestResult(docs=docs, page_texts=page_texts, images=images, page_blocks=page_blocks)


def _maybe_create_ocr(*, ocr_enabled: bool, language: str) -> Optional[object]:
    if not ocr_enabled:
        return None
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        console.print(
            "[yellow]OCR enabled but PaddleOCR is not installed. "
            "Install paddleocr/paddlepaddle then rerun.[/yellow]"
        )
        return None

    # 企业文档更关注中文；PaddleOCR 内部会自动处理角度/检测/识别
    return PaddleOCR(use_angle_cls=True, lang=language)


def _run_ocr(ocr: object, image_path: Path) -> str | None:
    try:
        # PaddleOCR.ocr 返回结构：[[[box], (text, score)], ...]
        result = getattr(ocr, "ocr")(str(image_path), cls=True)
        if not result:
            return None
        lines: list[str] = []
        for row in result:
            if not row:
                continue
            for item in row:
                if not item or len(item) < 2:
                    continue
                txt = item[1][0]
                if txt:
                    lines.append(str(txt))
        joined = "\n".join(lines).strip()
        return joined or None
    except Exception:
        return None


def _render_page_to_jpg(*, page: fitz.Page, out_path: Path, dpi: int = 200) -> None:
    dpi = max(72, int(dpi))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))


