# 企业文档多模态 RAG 架构文档

## 数据流全景图

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           企业文档多模态 RAG 数据流架构                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════
 阶段 1: INGEST（导入）                                          mrag/ingest/pdf.py
═══════════════════════════════════════════════════════════════════════════════════════════

                    ┌──────────────┐
                    │   PDF 文件    │
                    │ (企业文档)    │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │      PyMuPDF 解析       │
              │   fitz.open(pdf_path)  │
              └────────────┬───────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐
    │  文本提取    │ │  图片提取    │ │ 扫描件检测       │
    │ page.get_   │ │ page.get_   │ │ len(text)<阈值   │
    │ text()      │ │ images()    │ │ → 渲染整页       │
    └──────┬──────┘ └──────┬──────┘ └────────┬────────┘
           │               │                  │
           │               │                  ▼
           │               │         ┌─────────────────┐
           │               │         │   PaddleOCR     │
           │               │         │  (可选，中英)    │
           │               │         └────────┬────────┘
           │               │                  │
           ▼               ▼                  ▼
    ┌──────────────────────────────────────────────────┐
    │              落盘 (mrag/storage/manifest.py)      │
    │  docs.jsonl  |  page_texts.jsonl  |  images.jsonl │
    │              └── images/*.png (图片文件)          │
    └──────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
 阶段 2: INDEX（建索引）                                         mrag/cli.py → index()
═══════════════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────┐                          ┌─────────────────┐
    │ page_texts.jsonl│                          │  images.jsonl   │
    │    (文本)       │                          │  + images/*.png │
    └────────┬────────┘                          └────────┬────────┘
             │                                            │
             ▼                                            ▼
    ┌─────────────────┐                          ┌─────────────────┐
    │  切块 Chunking   │                          │                 │
    │ simple_char_    │                          │                 │
    │ chunk()         │                          │                 │
    │ (mrag/ingest/   │                          │                 │
    │  chunking.py)   │                          │                 │
    └────────┬────────┘                          │                 │
             │                                   │                 │
             ▼                                   ▼                 │
    ┌─────────────────┐                 ┌─────────────────┐        │
    │  TextEmbedder   │                 │  ImageEmbedder  │        │
    │  (BGE-M3 等)    │                 │  (CLIP/SigLIP)  │        │
    │ mrag/models/    │                 │ mrag/models/    │        │
    │ text_embed.py   │                 │ image_embed.py  │        │
    └────────┬────────┘                 └────────┬────────┘        │
             │                                   │                 │
             │  encode()                         │ encode_images() │
             ▼                                   ▼                 │
    ┌─────────────────┐                 ┌─────────────────┐        │
    │   text.faiss    │                 │  image.faiss    │        │
    │   text_meta.pkl │                 │  image_meta.pkl │        │
    │ (mrag/index/    │                 │ (mrag/index/    │        │
    │  faiss_text.py) │                 │  faiss_image.py)│        │
    └─────────────────┘                 └─────────────────┘        │
             │                                   │                 │
             └───────────────┬───────────────────┘                 │
                             │                                     │
                             ▼                                     │
                    ┌─────────────────┐                            │
                    │   index/ 目录    │◄───────────────────────────┘
                    │ (双索引落盘)     │
                    └─────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
 阶段 3: QUERY（检索 + 融合）                                    mrag/cli.py → query()
═══════════════════════════════════════════════════════════════════════════════════════════

                         ┌──────────────┐
                         │   用户问题    │
                         │  "报销流程    │
                         │  需要什么？"  │
                         └──────┬───────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
       ┌─────────────────┐             ┌─────────────────┐
       │  TextEmbedder   │             │  ImageEmbedder  │
       │   .encode()     │             │  .encode_text() │  ◄── 多模态关键点 1
       │  问题→文本向量   │             │  问题→图像空间   │      "以文搜图"
       └────────┬────────┘             └────────┬────────┘
                │                               │
                ▼                               ▼
       ┌─────────────────┐             ┌─────────────────┐
       │  search_text()  │             │ search_image()  │
       │  text.faiss     │             │  image.faiss    │
       └────────┬────────┘             └────────┬────────┘
                │                               │
                │  TextHit[]                    │  ImageHit[]
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   fuse_hits()   │  ◄── 多模态关键点 2
                       │ mrag/retrieval/ │      文本证据 + 图片证据融合
                       │ hybrid.py       │
                       │                 │
                       │ score = α*text  │
                       │ + (1-α)*image   │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Evidence[]    │
                       │ type: text|image│
                       │ doc_id, page,   │
                       │ score, text,    │
                       │ image_path ...  │
                       └────────┬────────┘
                                │
                                ▼


═══════════════════════════════════════════════════════════════════════════════════════════
 阶段 4: MLLM 生成（可选）                                       mrag/models/qwen2_vl.py
═══════════════════════════════════════════════════════════════════════════════════════════

                       ┌─────────────────┐
                       │   Evidence[]    │
                       └────────┬────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
       ┌─────────────────┐             ┌─────────────────┐
       │  evidence_texts │             │ evidence_images │
       │  (文本片段)      │             │  (图片路径)      │
       └────────┬────────┘             └────────┬────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │         Qwen2-VL              │  ◄── 多模态关键点 3
                │  answer_with_evidence()       │      同时输入 text + images
                │                               │
                │  ┌─────────────────────────┐  │
                │  │ processor(              │  │
                │  │   text=prompt,          │  │
                │  │   images=[img1,img2...] │  │  ◄── 真正的多模态输入
                │  │ )                       │  │
                │  └─────────────────────────┘  │
                │                               │
                │  model.generate(**inputs)     │
                └───────────────┬───────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     答案        │
                       │ + 引用(doc_id,  │
                       │   page_index,   │
                       │   image_id)     │
                       └─────────────────┘
```

---

## 文件对照表

| 阶段 | 文件 / 函数 |
|------|-------------|
| CLI 入口 | `mrag/cli.py` (ingest / index / query) |
| PDF解析 + OCR | `mrag/ingest/pdf.py` → `ingest_pdfs()` |
| 文本切块 | `mrag/ingest/chunking.py` → `rule_chunk()`（默认）/ `simple_char_chunk()` |
| 文本 Embedding | `mrag/models/text_embed.py` → `TextEmbedder` |
| 图像 Embedding (CLIP) | `mrag/models/image_embed.py` → `ImageEmbedder` |
| FAISS 文本索引 | `mrag/index/faiss_text.py` |
| FAISS 图像索引 | `mrag/index/faiss_image.py` |
| 检索融合 (多模态证据) | `mrag/retrieval/hybrid.py` → `fuse_hits()` |
| MLLM 生成 (Qwen2-VL) | `mrag/models/qwen2_vl.py` → `answer_with_evidence()` |
| Streamlit UI | `app/streamlit_app.py` |
| 评测脚本 | `eval/run_eval.py` |

---

## 三个"多模态关键点"

### 关键点 1：以文搜图（Text-to-Image Retrieval）

**位置**：`mrag/models/image_embed.py` → `ImageEmbedder.encode_text()`

**原理**：CLIP/SigLIP 模型将文本和图像编码到**同一个向量空间**。用户的文本问题可以直接编码后，在图像索引里做最近邻搜索，找到语义相关的图片。

```python
# 文本问题编码到 CLIP 图像空间
qv_img = ie.encode_text([question])[0]
image_hits = search_image(index=i_index, images=i_assets, query_vec=qv_img, top_k=5)
```

### 关键点 2：多模态证据融合（Hybrid Retrieval Fusion）

**位置**：`mrag/retrieval/hybrid.py` → `fuse_hits()`

**原理**：把文本检索结果（TextHit）和图像检索结果（ImageHit）统一成 `Evidence` 类型，按加权分数排序后返回 TopK。

```python
# 融合公式
text_score_normalized = α * minmax(text_score)
image_score_normalized = (1-α) * minmax(image_score)
```

**配置**：`configs/default.yaml` 里的 `retrieval.score_alpha`
- `α=0.7`：偏重文本证据
- `α=0.3`：偏重图片证据

### 关键点 3：多模态生成（Multimodal Generation）

**位置**：`mrag/models/qwen2_vl.py` → `Qwen2VL.answer_with_evidence()`

**原理**：Qwen2-VL 是真正的多模态大模型，可以在**同一次 forward** 中同时接收文本和图像输入，生成基于多模态证据的答案。

```python
# 同时输入 text + images
inputs = self._processor(
    text=prompt,
    images=images if images else None,
    return_tensors="pt"
)
out_ids = self._model.generate(**inputs, max_new_tokens=512)
```

---

## 企业场景推荐增强：文本 Reranker（强烈建议）

### 作用
企业制度/流程/条款问答中，向量召回容易把“看起来相似但不回答问题”的段落排在前面。Reranker 用交叉编码器对 `(query, passage)` 逐对打分，可显著提升 TopK 的精度与引用稳定性。

### 位置
- 配置：`configs/default.yaml` → `retrieval.text_rerank.*`
- 代码：`mrag/models/rerank.py`（`TextReranker`）
- 接入：`mrag/cli.py` 与 `app/streamlit_app.py`（query 时对文本 hit 重排）

### 开启方式
把 `configs/default.yaml` 改为：

```yaml
retrieval:
  text_rerank:
    enabled: true
    model: "BAAI/bge-reranker-v2-m3"
    top_n: 30
```

## 学习路径建议

---

## B1.1：跨页桥接切块（Cross-Page Bridge Chunks）

### 为什么需要
企业制度/流程/合同条款经常跨页（尤其是编号列表、长段落、表格说明）。如果严格按页切块，会导致：
- 条款被截断，向量召回命中率下降
- 引用不稳定（上一页命中不到，下一页缺少上文）

### 我们的做法
在每个相邻页之间，额外生成一个 **bridge chunk**：

```
tail(page_i) + [PAGE_BREAK] + head(page_{i+1})
```

它会作为额外文本块参与 `text.faiss` 检索，并在证据里标注页跨度 `page_index -> page_end`。

### 配置项
`configs/default.yaml`：

```yaml
chunking:
  cross_page:
    enabled: true
    tail_chars: 450
    head_chars: 450
    min_chars: 200
```

### 代码位置
- `mrag/ingest/chunking.py`：`cross_page_bridge_chunks()` + `TextChunk.page_end`
- `mrag/cli.py`：`index()` 构建时为每个 doc 的相邻页生成 bridge chunks

---

## B2：引用定位可视化（Evidence Grounding Visualization）

### 轻量版（已实现）
在 Streamlit 展示证据时：
- **text evidence**：展示其对应 `doc_id + page_index` 的整页渲染图（跨页证据展示两页）
- **image evidence**：展示图片本身（page/embedded 均可）

### 启用条件
需要 ingestion 阶段生成整页渲染图：

```yaml
ingest:
  render_pages:
    enabled: true
```

### 代码位置
- `app/streamlit_app.py`：构建 (doc_id,page_index)->page_image_path 映射并展示
- `mrag/ingest/pdf.py`：生成 `assets/pages/*.jpg`（kind=page）

### 高亮版（未来增强）
要在页图上“框出”证据具体位置，需要在 ingestion 记录文本块的 bbox（或引入版面分析/文本定位）。

### 高亮版（已实现：block bbox 级别）
当前实现方式：
- ingestion 阶段保存 `page_blocks.jsonl`：每页的文本块（blocks）bbox + 文本
- index 阶段把 blocks 的 bbox 通过字符串匹配绑定到 chunk（`TextChunk.bboxes`）
- Streamlit 展示页图时，将 bbox 按页尺寸缩放后画红框

局限：
- 这是**块级别**高亮，不是逐字逐行对齐；对 OCR 文本或强格式化 PDF 可能不够精确

### 第一步：跑通 ingest + index + query（不开 MLLM）

体会"多模态检索"是怎么把文字问题同时找回文本证据和图片证据的。

```bash
# 1. 导入 PDF
python -m mrag.cli ingest --input_dir /path/to/pdfs --dataset enterprise_docs

# 2. 建索引
python -m mrag.cli index --dataset enterprise_docs

# 3. 查询（此时 mllm.enabled=false，只返回证据）
python -m mrag.cli query --dataset enterprise_docs --question "报销流程需要哪些材料？"
```

### 第二步：打开 MLLM 生成

修改 `configs/default.yaml`：

```yaml
models:
  mllm:
    enabled: true
    name: "Qwen/Qwen2-VL-7B-Instruct"
```

再次运行 query，观察 Qwen2-VL 怎么同时吃 text + images 生成答案。

### 第三步：调整融合权重

修改 `configs/default.yaml`：

```yaml
retrieval:
  score_alpha: 0.5  # 调大偏重文本，调小偏重图片
```

### 第四步：加自己的 PDF

观察：
- OCR 是否被触发（扫描件/图片型 PDF）
- 图片是否被正确裁切
- 证据引用是否准确

---

## 目录结构

```
multimodal_ai/
├── app/
│   └── streamlit_app.py      # Web UI
├── configs/
│   └── default.yaml          # 配置文件
├── data/
│   └── {dataset}/            # 数据集目录
│       ├── docs.jsonl
│       ├── page_texts.jsonl
│       ├── images.jsonl
│       ├── images/           # 提取的图片
│       └── index/            # FAISS 索引
├── docs/
│   └── architecture.md       # 本文档
├── eval/
│   ├── questions.example.jsonl
│   └── run_eval.py           # 评测脚本
├── models/                   # 模型缓存（可选）
├── mrag/
│   ├── __init__.py
│   ├── cli.py                # 命令行入口
│   ├── config.py             # 配置加载
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── chunking.py       # 文本切块
│   │   └── pdf.py            # PDF 解析 + OCR
│   ├── index/
│   │   ├── __init__.py
│   │   ├── faiss_image.py    # 图像索引
│   │   └── faiss_text.py     # 文本索引
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_embed.py    # CLIP/SigLIP 图像编码
│   │   ├── qwen2_vl.py       # Qwen2-VL 多模态生成
│   │   └── text_embed.py     # BGE-M3 文本编码
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── hybrid.py         # 多模态融合
│   └── storage/
│       ├── __init__.py
│       └── manifest.py       # 数据落盘
├── requirements.txt
└── README.md
```

