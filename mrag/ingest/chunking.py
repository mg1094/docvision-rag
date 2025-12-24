from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    page_index: int
    # for cross-page chunks; when None, page_end == page_index
    page_end: int | None = None
    # B2：文本证据在 page 渲染图上的高亮框（PDF 坐标，points）
    bboxes: list[list[float]] | None = None
    # for cross-page chunks: bboxes on page_end
    bboxes_end: list[list[float]] | None = None
    text: str


def simple_char_chunk(
    doc_id: str,
    page_index: int,
    text: str,
    target_chars: int = 900,
    overlap_chars: int = 150,
) -> list[TextChunk]:
    t = (text or "").strip()
    if not t:
        return []
    if target_chars <= 0:
        raise ValueError("target_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be < target_chars")

    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    while start < len(t):
        end = min(len(t), start + target_chars)
        chunk_text = t[start:end].strip()
        if chunk_text:
            chunk_id = f"{doc_id}:p{page_index}:c{idx}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_index=page_index,
                    page_end=None,
                    text=chunk_text,
                )
            )
            idx += 1
        if end >= len(t):
            break
        start = max(0, end - overlap_chars)
    return chunks


_RE_HEADING = re.compile(
    r"^("
    r"(第[一二三四五六七八九十百千0-9]+[章节条款部分])"
    r"|(\d{1,3}(\.\d{1,3}){1,3})"
    r"|([（(]?\d+[)）]?)"
    r"|([一二三四五六七八九十]+[、.])"
    r"|([0-9]+[、.)])"
    r")"
)


def _normalize_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # 常见：多空行
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # 太长通常不是标题/条款编号
    if len(s) > 80:
        return False
    return bool(_RE_HEADING.match(s))


def rule_chunk(
    *,
    doc_id: str,
    page_index: int,
    text: str,
    target_chars: int = 900,
    overlap_chars: int = 150,
    min_chunk_chars: int = 120,
) -> list[TextChunk]:
    """
    企业文档友好切块：
    - 优先按“标题/章/条/编号列表”断开（避免切断条款）
    - 再在每个 section 内按段落边界做长度控制（必要时退化为字符滑窗）
    """
    t = _normalize_text(text)
    if not t:
        return []
    if target_chars <= 0:
        raise ValueError("target_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be < target_chars")

    # 1) 先按行切出“章节/条款”section
    lines = [ln.strip() for ln in t.split("\n")]
    sections: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if _is_heading(ln) and cur:
            sections.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append(cur)

    # 2) 每个 section 再按段落边界组合到 target_chars
    chunks: list[TextChunk] = []
    chunk_idx = 0

    def flush(buf: list[str], sec_i: int) -> None:
        nonlocal chunk_idx
        txt = "\n".join([x for x in buf if x is not None]).strip()
        if len(txt) < max(1, int(min_chunk_chars)):
            return
        chunk_id = f"{doc_id}:p{page_index}:s{sec_i}:c{chunk_idx}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page_index=page_index,
                page_end=None,
                text=txt,
            )
        )
        chunk_idx += 1

    for sec_i, sec_lines in enumerate(sections):
        # split paragraphs by blank lines (already stripped, so blank line is "")
        paras: list[str] = []
        buf: list[str] = []
        for ln in sec_lines:
            if ln == "":
                if buf:
                    paras.append("\n".join(buf).strip())
                    buf = []
                continue
            buf.append(ln)
        if buf:
            paras.append("\n".join(buf).strip())

        # greedy pack paragraphs
        cur_buf: list[str] = []
        for p in paras:
            if not cur_buf:
                cur_buf = [p]
                continue
            cand = "\n\n".join(cur_buf + [p])
            if len(cand) <= target_chars:
                cur_buf.append(p)
            else:
                flush(cur_buf, sec_i)
                cur_buf = [p]

        if cur_buf:
            # 若最后一块仍太长，则退化为字符滑窗
            joined = "\n\n".join(cur_buf).strip()
            if len(joined) <= target_chars:
                flush(cur_buf, sec_i)
            else:
                # fallback: keep section id prefix
                sub_chunks = simple_char_chunk(
                    doc_id=doc_id,
                    page_index=page_index,
                    text=joined,
                    target_chars=target_chars,
                    overlap_chars=overlap_chars,
                )
                for sc in sub_chunks:
                    # rewrite chunk_id to include section
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{doc_id}:p{page_index}:s{sec_i}:{sc.chunk_id.split(':')[-1]}",
                            doc_id=doc_id,
                            page_index=page_index,
                            page_end=None,
                            text=sc.text,
                        )
                    )

    # 3) 若规则切块结果很少（比如无明显标题），退化为字符切块
    if len(chunks) <= 1:
        return simple_char_chunk(
            doc_id=doc_id,
            page_index=page_index,
            text=t,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
    return chunks


def cross_page_bridge_chunks(
    *,
    doc_id: str,
    left_page_index: int,
    left_text: str,
    right_page_index: int,
    right_text: str,
    tail_chars: int = 450,
    head_chars: int = 450,
    min_chars: int = 200,
) -> list[TextChunk]:
    """
    生成跨页“桥接 chunk”：上一页末尾 + 下一页开头。
    目的：覆盖跨页条款/列表导致的检索断裂。
    """
    lt = _normalize_text(left_text)
    rt = _normalize_text(right_text)
    if not lt or not rt:
        return []
    tail_chars = max(1, int(tail_chars))
    head_chars = max(1, int(head_chars))
    min_chars = max(1, int(min_chars))

    bridge = (lt[-tail_chars:] + "\n\n[PAGE_BREAK]\n\n" + rt[:head_chars]).strip()
    if len(bridge) < min_chars:
        return []

    chunk_id = f"{doc_id}:p{left_page_index}-{right_page_index}:bridge"
    return [
        TextChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_index=left_page_index,
            page_end=right_page_index,
            bboxes=None,
            bboxes_end=None,
            text=bridge,
        )
    ]


def attach_bboxes_to_chunks(
    chunks: list[TextChunk],
    *,
    blocks_by_page: dict[tuple[str, int], list[dict]],
    max_boxes: int = 20,
    match_prefix_chars: int = 24,
) -> list[TextChunk]:
    """
    将 page_blocks 的 bbox 绑定到 chunk 上，供 B2 在页图上画框。
    blocks_by_page[(doc_id,page_index)] = [{"bbox":[...],"text":"..."}, ...]
    """

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", (s or ""))

    def match_boxes(chunk_text: str, page_key: tuple[str, int]) -> list[list[float]]:
        bs = blocks_by_page.get(page_key, [])
        if not bs:
            return []
        ct = norm(chunk_text)
        out: list[list[float]] = []
        for b in bs:
            bt = str(b.get("text", "")).strip()
            if not bt:
                continue
            needle = norm(bt)[: max(6, int(match_prefix_chars))]
            if needle and needle in ct:
                bb = b.get("bbox")
                if isinstance(bb, list) and len(bb) == 4:
                    out.append([float(x) for x in bb])
            if len(out) >= max(1, int(max_boxes)):
                break
        return out

    out_chunks: list[TextChunk] = []
    for c in chunks:
        pe = c.page_end if c.page_end is not None else c.page_index
        if pe == c.page_index:
            b0 = match_boxes(c.text, (c.doc_id, int(c.page_index)))
            out_chunks.append(
                TextChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    page_index=c.page_index,
                    page_end=c.page_end,
                    bboxes=b0 or None,
                    bboxes_end=None,
                    text=c.text,
                )
            )
        else:
            left, right = c.text.split("[PAGE_BREAK]", 1) if "[PAGE_BREAK]" in c.text else (c.text, "")
            b0 = match_boxes(left, (c.doc_id, int(c.page_index)))
            b1 = match_boxes(right, (c.doc_id, int(pe)))
            out_chunks.append(
                TextChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    page_index=c.page_index,
                    page_end=pe,
                    bboxes=b0 or None,
                    bboxes_end=b1 or None,
                    text=c.text,
                )
            )
    return out_chunks


