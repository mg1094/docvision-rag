from __future__ import annotations

from pathlib import Path

import streamlit as st

from mrag.config import load_config
from mrag.index.faiss_image import load_image_index, search_image
from mrag.index.faiss_text import load_text_index, search_text
from mrag.models.image_embed import ImageEmbedder
from mrag.models.qwen2_vl import Qwen2VL
from mrag.models.rerank import TextReranker
from mrag.models.text_embed import TextEmbedder
from mrag.retrieval.hybrid import fuse_hits
from mrag.storage.manifest import load_page_blocks, load_records

st.set_page_config(page_title="企业文档多模态RAG（本地开源）", layout="wide")

st.title("企业文档多模态 RAG（本地开源）")
st.caption("本地开源：企业PDF/图片/扫描件 → 双索引检索 → 引用 →（可选）Qwen2-VL 生成。")


@st.cache_resource
def _load_cfg():
    return load_config("configs/default.yaml").raw


@st.cache_resource
def _load_text_components(index_dir: str):
    idx, chunks = load_text_index(Path(index_dir))
    return idx, chunks


@st.cache_resource
def _load_image_components(index_dir: str):
    idx, assets = load_image_index(Path(index_dir))
    return idx, assets


@st.cache_resource
def _text_embedder(model_name: str, batch_size: int):
    return TextEmbedder(model_name=model_name, batch_size=batch_size)


@st.cache_resource
def _image_embedder(model_name: str, batch_size: int):
    return ImageEmbedder(model_name=model_name, batch_size=batch_size)


@st.cache_resource
def _qwen2_vl(model_name: str, max_new_tokens: int):
    return Qwen2VL(model_name=model_name, max_new_tokens=max_new_tokens)


@st.cache_resource
def _page_image_map(dataset_dir: str) -> dict[tuple[str, int], str]:
    """
    Map (doc_id, page_index) -> rendered page image path.
    Requires ingest.render_pages.enabled=true (or OCR fallback generated pages).
    """
    ds = Path(dataset_dir)
    _, _, images = load_records(ds)
    m: dict[tuple[str, int], str] = {}
    for im in images:
        if getattr(im, "kind", None) == "page":
            m[(im.doc_id, int(im.page_index))] = im.path
    return m


@st.cache_resource
def _page_geom_map(dataset_dir: str) -> dict[tuple[str, int], tuple[float, float]]:
    """
    Map (doc_id,page_index) -> (page_width_pt, page_height_pt) for bbox scaling.
    """
    ds = Path(dataset_dir)
    m: dict[tuple[str, int], tuple[float, float]] = {}
    for pb in load_page_blocks(ds):
        m[(pb.doc_id, int(pb.page_index))] = (float(pb.page_width), float(pb.page_height))
    return m


def _draw_boxes_on_page(
    image_path: str,
    *,
    boxes_pt: list[list[float]] | None,
    page_w_pt: float | None,
    page_h_pt: float | None,
):
    if not boxes_pt or not page_w_pt or not page_h_pt:
        return None
    try:
        from PIL import Image, ImageDraw

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        sx = w / float(page_w_pt)
        sy = h / float(page_h_pt)
        for bb in boxes_pt[:30]:
            if not bb or len(bb) != 4:
                continue
            x0, y0, x1, y1 = [float(x) for x in bb]
            r = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
            draw.rectangle(r, outline=(255, 0, 0), width=4)
        return img
    except Exception:
        return None

with st.sidebar:
    st.header("数据集")
    cfg = _load_cfg()
    data_dir = Path(cfg["paths"]["data_dir"])
    dataset = st.text_input("dataset 名称", value=str(cfg["dataset"]["name"]))
    ds_dir = data_dir / dataset
    st.write("dataset 路径：", str(ds_dir.resolve()))
    index_dir = ds_dir / "index"
    st.write("index 路径：", str(index_dir.resolve()))

    st.divider()
    st.subheader("可选：MLLM")
    enable_mllm = st.checkbox("启用 Qwen2-VL 生成", value=bool(cfg["models"]["mllm"]["enabled"]))

st.subheader("0) 构建索引（只需一次）")
st.code(
    "python -m mrag.cli ingest --input_dir /ABS/PATH/TO/PDFS --dataset enterprise_docs\n"
    "python -m mrag.cli index --dataset enterprise_docs",
    language="bash",
)

st.subheader("1) 提问")
q = st.text_input("问题", value="报销流程需要哪些材料？")
if st.button("检索并回答", type="primary"):
    if not (index_dir / "text.faiss").exists():
        st.error("找不到文本索引：请先运行 CLI ingest + index 构建索引。")
        st.stop()

    t_index, t_chunks = _load_text_components(str(index_dir))
    te = _text_embedder(
        str(cfg["models"]["text_embedding"]["name"]),
        int(cfg["models"]["text_embedding"]["batch_size"]),
    )
    qv = te.encode([q])[0]
    text_hits = search_text(
        index=t_index,
        chunks=t_chunks,
        query_vec=qv,
        top_k=int(cfg["retrieval"]["top_k_text"]),
    )
    rr_cfg = cfg.get("retrieval", {}).get("text_rerank", {})
    if bool(rr_cfg.get("enabled")) and text_hits:
        top_n = int(rr_cfg.get("top_n", 30))
        cand = text_hits[: max(1, top_n)]
        reranker = TextReranker(model_name=str(rr_cfg.get("model")))
        scores = reranker.score(q, [h.text for h in cand])
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
        text_hits = reranked + text_hits[len(cand) :]

    image_hits = []
    if (index_dir / "image.faiss").exists():
        i_index, i_assets = _load_image_components(str(index_dir))
        ie = _image_embedder(
            str(cfg["models"]["image_embedding"]["name"]),
            int(cfg["models"]["image_embedding"]["batch_size"]),
        )
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

    st.subheader("答案（可选：Qwen2-VL）")
    if enable_mllm:
        try:
            qwen = _qwen2_vl(
                str(cfg["models"]["mllm"]["name"]),
                int(cfg["models"]["mllm"]["max_new_tokens"]),
            )
            ev_texts = [e.text for e in evs if e.type == "text" and e.text][:8]
            ev_imgs = [e.image_path for e in evs if e.type == "image" and e.image_path][:6]
            ans = qwen.answer_with_evidence(
                question=q,
                evidence_texts=ev_texts,
                evidence_images=ev_imgs,
            )
            st.write(ans)
        except Exception as e:
            st.error(f"MLLM 生成失败：{e}")
            st.info("你也可以先关闭 MLLM，只看检索证据与引用。")
    else:
        st.info("当前未启用 Qwen2-VL；下面展示检索到的证据与引用。")

    st.subheader("证据 / 引用（Top）")
    page_map = _page_image_map(str(ds_dir)) if ds_dir.exists() else {}
    geom_map = _page_geom_map(str(ds_dir)) if ds_dir.exists() else {}
    for i, e in enumerate(evs, start=1):
        kind = f" | kind={e.image_kind}" if getattr(e, "image_kind", None) else ""
        span = f"{e.page_index}-{e.page_end}" if getattr(e, "page_end", None) is not None else f"{e.page_index}"
        title = f"{i}. {e.type}{kind} | {e.doc_id} | page={span} | score={e.score:.3f}"
        with st.expander(title, expanded=(i <= 2)):
            if e.type == "text":
                # B2 轻量版：展示证据所在页的渲染图（若开启 render_pages）
                p0 = page_map.get((e.doc_id, int(e.page_index)))
                p1 = None
                if getattr(e, "page_end", None) is not None:
                    p1 = page_map.get((e.doc_id, int(e.page_end)))

                cols = st.columns(2) if p1 else st.columns(1)
                if p0 and Path(p0).exists():
                    pw0, ph0 = geom_map.get((e.doc_id, int(e.page_index)), (None, None))
                    img0 = _draw_boxes_on_page(
                        p0, boxes_pt=getattr(e, "text_bboxes", None), page_w_pt=pw0, page_h_pt=ph0
                    )
                    cols[0].image(
                        img0 if img0 is not None else p0,
                        caption=f"{e.doc_id} page={e.page_index}",
                        use_container_width=True,
                    )
                else:
                    cols[0].caption("未找到对应页渲染图：请在配置中开启 ingest.render_pages.enabled")

                if p1:
                    if p1 and Path(p1).exists():
                        pw1, ph1 = geom_map.get((e.doc_id, int(e.page_end)), (None, None))
                        img1 = _draw_boxes_on_page(
                            p1,
                            boxes_pt=getattr(e, "text_bboxes_end", None),
                            page_w_pt=pw1,
                            page_h_pt=ph1,
                        )
                        cols[1].image(
                            img1 if img1 is not None else p1,
                            caption=f"{e.doc_id} page={e.page_end}",
                            use_container_width=True,
                        )
                    else:
                        cols[1].caption("未找到跨页的第二页渲染图")

                st.write(e.text or "")
            else:
                if e.image_path and Path(e.image_path).exists():
                    st.image(e.image_path, caption=f"{e.image_id} ({e.image_kind})", use_container_width=True)
                if e.image_ocr_text:
                    st.caption("OCR（可选）")
                    st.text(e.image_ocr_text)


