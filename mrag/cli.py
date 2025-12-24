from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.pretty import pprint
from tqdm import tqdm

from mrag.config import load_config
from mrag.ingest.chunking import (
    TextChunk,
    attach_bboxes_to_chunks,
    cross_page_bridge_chunks,
    rule_chunk,
    simple_char_chunk,
)
from mrag.ingest.pdf import ingest_pdfs
from mrag.index.faiss_image import build_image_index, load_image_index, search_image
from mrag.index.faiss_text import build_text_index, load_text_index, search_text
from mrag.models.image_embed import ImageEmbedder
from mrag.models.qwen2_vl import Qwen2VL
from mrag.models.rerank import TextReranker
from mrag.models.text_embed import TextEmbedder
from mrag.retrieval.hybrid import fuse_hits
from mrag.storage.manifest import dump_page_blocks, dump_records
from mrag.storage.manifest import load_page_blocks, load_records


app = typer.Typer(add_completion=False)
console = Console()


def _dataset_dir(cfg_path: Path, dataset: str | None) -> Path:
    cfg = load_config(cfg_path).raw
    data_dir = Path(cfg["paths"]["data_dir"])
    name = dataset or cfg["dataset"]["name"]
    return data_dir / name


@app.command()
def ingest(
    input_dir: Path = typer.Option(..., help="包含PDF的目录（绝对路径更稳）"),
    dataset: str | None = typer.Option(None, help="数据集名称（用于隔离存储）"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="配置文件路径"),
) -> None:
    cfg = load_config(config).raw
    ds_dir = _dataset_dir(config, dataset)
    ds_dir.mkdir(parents=True, exist_ok=True)

    res = ingest_pdfs(
        input_dir=input_dir,
        pdf_glob=cfg["ingest"]["pdf_glob"],
        dataset_dir=ds_dir,
        image_max_per_page=int(cfg["ingest"]["image_max_per_page"]),
        render_pages_enabled=bool(cfg["ingest"]["render_pages"]["enabled"]),
        render_pages_dpi=int(cfg["ingest"]["render_pages"]["dpi"]),
        ocr_enabled=bool(cfg["ingest"]["ocr"]["enabled"]),
        ocr_language=str(cfg["ingest"]["ocr"]["language"]),
        ocr_min_text_chars=int(cfg["ingest"]["ocr"]["min_text_chars"]),
        ocr_page_render_dpi=int(cfg["ingest"]["ocr"]["page_render_dpi"]),
    )
    dump_records(ds_dir, res.docs, res.page_texts, res.images)
    dump_page_blocks(ds_dir, res.page_blocks)
    console.print(f"[green]OK[/green] wrote manifests under: {ds_dir}")
    console.print("- docs.jsonl, page_texts.jsonl, images.jsonl, page_blocks.jsonl")


@app.command()
def inspect(
    dataset: str = typer.Option(..., help="数据集名称"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="配置文件路径"),
) -> None:
    ds_dir = _dataset_dir(config, dataset)
    console.print(f"[cyan]Inspect[/cyan] {ds_dir}")
    for name in ["docs.jsonl", "page_texts.jsonl", "images.jsonl"]:
        p = ds_dir / name
        console.print(f"- {name}: {'exists' if p.exists() else 'missing'}")


@app.command()
def index(
    dataset: str = typer.Option(..., help="数据集名称"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="配置文件路径"),
) -> None:
    cfg = load_config(config).raw
    ds_dir = _dataset_dir(config, dataset)
    index_dir = ds_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    _, pages, images = load_records(ds_dir)
    page_blocks = load_page_blocks(ds_dir)
    blocks_by_page: dict[tuple[str, int], list[dict]] = {}
    for pb in page_blocks:
        blocks_by_page[(pb.doc_id, int(pb.page_index))] = [
            {"bbox": b.bbox, "text": b.text} for b in pb.blocks
        ]

    # 1) chunk
    chunks: list[TextChunk] = []
    # group by doc for cross-page bridging
    pages_by_doc: dict[str, list] = {}
    for p in pages:
        pages_by_doc.setdefault(p.doc_id, []).append(p)

    for _, doc_pages in pages_by_doc.items():
        doc_pages.sort(key=lambda x: x.page_index)
        method = str(cfg.get("chunking", {}).get("method", "char")).lower()
        for p in doc_pages:
            if method == "rule":
                chunks.extend(
                    rule_chunk(
                        doc_id=p.doc_id,
                        page_index=p.page_index,
                        text=p.text,
                        target_chars=int(cfg["chunking"]["target_chars"]),
                        overlap_chars=int(cfg["chunking"]["overlap_chars"]),
                        min_chunk_chars=int(cfg["chunking"].get("min_chunk_chars", 120)),
                    )
                )
            else:
                chunks.extend(
                    simple_char_chunk(
                        doc_id=p.doc_id,
                        page_index=p.page_index,
                        text=p.text,
                        target_chars=int(cfg["chunking"]["target_chars"]),
                        overlap_chars=int(cfg["chunking"]["overlap_chars"]),
                    )
                )

        # B1.1 cross-page bridge chunks
        cp_cfg = cfg.get("chunking", {}).get("cross_page", {})
        if bool(cp_cfg.get("enabled", True)) and len(doc_pages) >= 2:
            for left, right in zip(doc_pages, doc_pages[1:], strict=False):
                chunks.extend(
                    cross_page_bridge_chunks(
                        doc_id=left.doc_id,
                        left_page_index=int(left.page_index),
                        left_text=left.text,
                        right_page_index=int(right.page_index),
                        right_text=right.text,
                        tail_chars=int(cp_cfg.get("tail_chars", 450)),
                        head_chars=int(cp_cfg.get("head_chars", 450)),
                        min_chars=int(cp_cfg.get("min_chars", 200)),
                    )
                )
    # B2 高亮：将 page_blocks 的 bbox 匹配回 chunk
    if blocks_by_page:
        chunks = attach_bboxes_to_chunks(chunks, blocks_by_page=blocks_by_page)

    console.print(f"[cyan]Chunks[/cyan]: {len(chunks)}")

    # 2) text embedding + faiss
    if chunks:
        te = TextEmbedder(
            model_name=str(cfg["models"]["text_embedding"]["name"]),
            batch_size=int(cfg["models"]["text_embedding"]["batch_size"]),
        )
        texts = [c.text for c in chunks]
        text_embs = te.encode(tqdm(texts, desc="text-embed"))
        build_text_index(out_dir=index_dir, chunks=chunks, embeddings=text_embs)
        console.print(f"[green]OK[/green] built text index: {index_dir / 'text.faiss'}")
    else:
        console.print("[yellow]No text chunks to index[/yellow]")

    # 3) image embedding + faiss
    if images:
        ie = ImageEmbedder(
            model_name=str(cfg["models"]["image_embedding"]["name"]),
            batch_size=int(cfg["models"]["image_embedding"]["batch_size"]),
        )
        paths = [im.path for im in images]
        img_embs = ie.encode_images(tqdm(paths, desc="image-embed"))
        build_image_index(out_dir=index_dir, images=images, embeddings=img_embs)
        console.print(f"[green]OK[/green] built image index: {index_dir / 'image.faiss'}")
    else:
        console.print("[yellow]No images to index[/yellow]")


@app.command()
def query(
    dataset: str = typer.Option(..., help="数据集名称"),
    question: str = typer.Option(..., help="你的问题"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="配置文件路径"),
) -> None:
    cfg = load_config(config).raw
    ds_dir = _dataset_dir(config, dataset)
    index_dir = ds_dir / "index"
    console.print(f"[cyan]Query[/cyan] dataset={ds_dir} question={question!r}")

    # load indexes (if exist)
    text_hits = []
    image_hits = []

    if (index_dir / "text.faiss").exists():
        t_index, t_chunks = load_text_index(index_dir)
        te = TextEmbedder(
            model_name=str(cfg["models"]["text_embedding"]["name"]),
            batch_size=int(cfg["models"]["text_embedding"]["batch_size"]),
        )
        qv = te.encode([question])[0]
        text_hits = search_text(
            index=t_index,
            chunks=t_chunks,
            query_vec=qv,
            top_k=int(cfg["retrieval"]["top_k_text"]),
        )
        # 可选：对文本召回结果做 rerank（更适合企业条款/流程）
        rr_cfg = cfg.get("retrieval", {}).get("text_rerank", {})
        if bool(rr_cfg.get("enabled")) and text_hits:
            top_n = int(rr_cfg.get("top_n", 30))
            cand = text_hits[: max(1, top_n)]
            reranker = TextReranker(model_name=str(rr_cfg.get("model")))
            scores = reranker.score(question, [h.text for h in cand])
            reranked = []
            for h, s in zip(cand, scores.tolist(), strict=False):
                reranked.append(
                    type(h)(
                        chunk_id=h.chunk_id,
                        doc_id=h.doc_id,
                        page_index=h.page_index,
                        score=float(s),
                        text=h.text,
                    )
                )
            reranked.sort(key=lambda x: x.score, reverse=True)
            # 只替换前 top_n 的顺序/分数，后面的保持原样（也可直接截断）
            text_hits = reranked + text_hits[len(cand) :]

    if (index_dir / "image.faiss").exists():
        i_index, i_assets = load_image_index(index_dir)
        ie = ImageEmbedder(
            model_name=str(cfg["models"]["image_embedding"]["name"]),
            batch_size=int(cfg["models"]["image_embedding"]["batch_size"]),
        )
        qv_img = ie.encode_text([question])[0]
        image_hits = search_image(
            index=i_index,
            images=i_assets,
            query_vec=qv_img,
            top_k=int(cfg["retrieval"]["top_k_image"]),
        )

    evs = fuse_hits(
        text_hits=text_hits,
        image_hits=image_hits,
        alpha=float(cfg["retrieval"]["score_alpha"]),
        fuse_k=int(cfg["retrieval"]["fuse_k"]),
    )

    # 最小输出：证据 + 引用（可选：接入 Qwen2-VL 生成）
    answer: str | None = None
    if bool(cfg["models"]["mllm"]["enabled"]):
        qwen = Qwen2VL(
            model_name=str(cfg["models"]["mllm"]["name"]),
            max_new_tokens=int(cfg["models"]["mllm"]["max_new_tokens"]),
        )
        ev_texts = [e.text for e in evs if e.type == "text" and e.text][:8]
        ev_imgs = [e.image_path for e in evs if e.type == "image" and e.image_path][:6]
        answer = qwen.answer_with_evidence(
            question=question,
            evidence_texts=ev_texts,
            evidence_images=ev_imgs,
        )

    out = {
        "question": question,
        "dataset_dir": str(ds_dir),
        "answer": answer,
        "evidence": [
            {
                "type": e.type,
                "doc_id": e.doc_id,
                "page_index": e.page_index,
                "page_end": e.page_end,
                "score": round(float(e.score), 4),
                "chunk_id": e.chunk_id,
                "text_preview": (e.text[:180] + "...") if e.text else None,
                "text_bboxes": e.text_bboxes,
                "text_bboxes_end": e.text_bboxes_end,
                "image_id": e.image_id,
                "image_path": e.image_path,
                "image_kind": e.image_kind,
                "image_ocr_preview": (e.image_ocr_text[:180] + "...") if e.image_ocr_text else None,
            }
            for e in evs
        ],
    }
    pprint(out)


def main() -> None:
    app()


if __name__ == "__main__":
    main()


