from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from mrag.config import load_config
from mrag.index.faiss_image import load_image_index, search_image
from mrag.index.faiss_text import load_text_index, search_text
from mrag.models.image_embed import ImageEmbedder
from mrag.models.text_embed import TextEmbedder
from mrag.retrieval.hybrid import fuse_hits


console = Console()


@dataclass(frozen=True)
class GoldRef:
    doc_id: str
    page_index: int


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _hit(golds: list[GoldRef], evs: list[dict], k: int) -> bool:
    top = evs[:k]
    for g in golds:
        for e in top:
            if e.get("doc_id") == g.doc_id and int(e.get("page_index")) == int(g.page_index):
                return True
    return False


def main() -> None:
    cfg = load_config("configs/default.yaml").raw
    dataset = cfg["dataset"]["name"]
    ds_dir = Path(cfg["paths"]["data_dir"]) / dataset
    index_dir = ds_dir / "index"

    q_path = Path("eval/questions.example.jsonl")
    rows = _read_jsonl(q_path)
    console.print(f"[cyan]Eval[/cyan] questions={len(rows)} dataset={dataset}")

    # load indexes + embedders once
    t_index, t_chunks = load_text_index(index_dir)
    i_index, i_assets = load_image_index(index_dir) if (index_dir / "image.faiss").exists() else (None, [])

    te = TextEmbedder(model_name=str(cfg["models"]["text_embedding"]["name"]))
    ie = ImageEmbedder(model_name=str(cfg["models"]["image_embedding"]["name"]))

    ks = [1, 3, 5, 10]
    hit_cnt = {k: 0 for k in ks}

    for r in rows:
        q = r["question"]
        golds = [GoldRef(**g) for g in r.get("gold_refs", [])]
        qv = te.encode([q])[0]
        text_hits = search_text(index=t_index, chunks=t_chunks, query_vec=qv, top_k=int(cfg["retrieval"]["top_k_text"]))

        image_hits = []
        if i_index is not None:
            qvi = ie.encode_text([q])[0]
            image_hits = search_image(
                index=i_index,
                images=i_assets,
                query_vec=qvi,
                top_k=int(cfg["retrieval"]["top_k_image"]),
            )

        evs = fuse_hits(
            text_hits=text_hits,
            image_hits=image_hits,
            alpha=float(cfg["retrieval"]["score_alpha"]),
            fuse_k=int(cfg["retrieval"]["fuse_k"]),
        )
        ev_dicts = [{"doc_id": e.doc_id, "page_index": e.page_index, "type": e.type} for e in evs]
        for k in ks:
            if _hit(golds, ev_dicts, k):
                hit_cnt[k] += 1

    for k in ks:
        rate = hit_cnt[k] / max(1, len(rows))
        console.print(f"Hit@{k}: {hit_cnt[k]}/{len(rows)} = {rate:.3f}")


if __name__ == "__main__":
    main()


