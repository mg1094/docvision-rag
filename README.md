# DocVision RAG - 企业文档多模态 RAG（本地开源）- 学习型一条龙项目

这个项目用于用**一个完整可运行的工程**学习多模态：企业文档（PDF/图片/扫描件）→ 文本/图片双索引 → 检索融合 → **Qwen2-VL 本地生成** → **答案带引用（页码/图块）** → 最小评测。

## 你将学到什么
- **企业文档 ingestion**：PDF 文本提取、图片/图表提取、扫描件 OCR（可选）
- **多模态索引**：文本 embedding（BGE-M3）+ 图片 embedding（CLIP/SigLIP）+ FAISS
- **检索融合**：把“文本证据 + 图片证据”合成一个可控的上下文
- **MLLM 约束生成**：强制模型只基于证据回答，并输出引用
- **评测**：检索命中率、引用正确率、人工打分闭环

## 目录结构
```
.
├── app/
│   └── streamlit_app.py          # Demo（上传/检索/问答/引用展示）
├── mrag/
│   ├── cli.py                    # 命令行：ingest / index / query
│   ├── config.py                 # 配置加载（YAML）
│   ├── ingest/
│   │   ├── pdf.py                # PDF/图片提取 + 可选OCR
│   │   └── chunking.py           # 文本切块
│   ├── index/
│   │   ├── faiss_text.py         # 文本向量索引
│   │   └── faiss_image.py        # 图片向量索引
│   ├── models/
│   │   ├── text_embed.py         # BGE-M3 embedding
│   │   ├── image_embed.py        # CLIP/SigLIP embedding
│   │   └── qwen2_vl.py           # Qwen2-VL 本地推理（可选）
│   ├── retrieval/
│   │   └── hybrid.py             # 文本/图片结果融合
│   └── storage/
│       └── manifest.py           # 元数据（JSONL）
├── eval/
│   ├── run_eval.py               # 最小评测脚本
│   └── questions.example.jsonl   # 评测格式示例
├── configs/
│   └── default.yaml              # 默认配置（模型/路径/参数）
├── data/
│   └── .gitkeep
├── pyproject.toml                # 项目配置和依赖（uv/pip 通用）
├── requirements.txt              # pip 兼容（可选，推荐用 pyproject.toml）
├── LICENSE                       # MIT 许可证
└── .gitignore
```

## 环境要求
- Python 3.10+
- **uv**（推荐）：快速 Python 包管理器，[安装指南](https://github.com/astral-sh/uv#installation)
- **算力充足**：建议 NVIDIA CUDA（Linux）体验最好；Mac 也能跑，但大模型推理可能较慢

## 安装（本地开源）

### 方式 1：使用 uv（推荐）

> 提示：本仓库不自动下载模型；首次运行会从 HuggingFace 缓存目录加载（请确保你的机器能访问镜像/内网仓库）。

```bash
# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖（会自动创建虚拟环境并安装所有包）
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

### 方式 2：使用 pip（兼容）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 安装 PyTorch（必须）
本项目的 embedding / reranker / Qwen2-VL 都依赖 PyTorch。由于不同平台/CUDA 版本安装命令不同，**没有把 torch 写进 pyproject.toml**。

- **NVIDIA CUDA（推荐）**：按你 CUDA 版本用官方命令安装
  ```bash
  # 例如 CUDA 11.8
  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **CPU**：安装 CPU 版 torch（速度较慢但可跑通）
  ```bash
  uv pip install torch torchvision torchaudio
  ```
- **Apple Silicon（MPS）**：可安装官方 Mac 版 torch（推理速度视模型而定）
  ```bash
  uv pip install torch torchvision torchaudio
  ```

安装完成后，可用 `python -c "import torch; print(torch.__version__)"` 验证。

## 快速开始

### 1) 写入文档（构建索引）
```bash
python -m mrag.cli ingest --input_dir /ABS/PATH/TO/PDFS --dataset enterprise_docs
```

### 1.5) （推荐）开启整页渲染（表格/图表/B2 可视化需要）
编辑 `configs/default.yaml`：
- `ingest.render_pages.enabled: true`

### 2) 构建索引（必须先做一次）
```bash
python -m mrag.cli index --dataset enterprise_docs
```

### 3) 命令行提问（先走检索+引用；MLLM 可选）
```bash
python -m mrag.cli query --dataset enterprise_docs --question "报销流程需要哪些材料？"
```

### 4) 启动 Demo
```bash
streamlit run app/streamlit_app.py
```

## 最小评测
1) 先准备问题集（见 `eval/questions.example.jsonl`）
2) 确保已经 `ingest` + `index`
3) 运行：

```bash
python eval/run_eval.py
```

## 重要设计点（企业文档）
- **引用优先**：答案必须带 “页码 + 证据片段/图块 id”，否则判定为低可信
- **扫描件优先策略**：当 PDF 抽取文本为空/很少时，自动走 OCR
- **矢量表格/图表兜底**：可开启“整页渲染入图像索引”，让以文搜图能命中表格/图表页（见 `configs/default.yaml` 的 `ingest.render_pages.enabled`）
- **文本 reranker（推荐）**：可开启 `retrieval.text_rerank.enabled`，对召回的文本 TopN 重排，企业制度/流程类效果通常更稳
- **结构化切块（推荐）**：默认 `chunking.method: rule`，优先按“章/条/编号列表”切，再做长度控制，减少切断条款导致的引用不准
- **跨页桥接切块（B1.1）**：默认开启 `chunking.cross_page.enabled`，自动生成“上一页尾 + 下一页头”的 bridge chunk，覆盖跨页条款/列表
- **B2 引用可视化（已实现轻量版）**：在 Streamlit 中对文本证据展示对应页渲染图；需要开启 `ingest.render_pages.enabled: true`
- **B2 高亮版（已实现）**：在 Streamlit 中把命中的文本证据块画在页图上（红框）；需要重新 `ingest` 生成 `page_blocks.jsonl`，并重新 `index`
- **数据隔离**：每个 dataset 独立存储（便于权限/审计/迭代）

## 常见问题（提交/复现）
- **为什么 import 报错缺少 typer/streamlit 等？**：请先 `uv sync`（或 `pip install -r requirements.txt`）安装依赖
- **为什么模型下载很慢/失败？**：需要配置 HuggingFace 镜像/缓存（企业内网常见）；也可提前离线下载到缓存目录
- **改了切块/开启了 B2 高亮后为什么效果没变？**：这类改动需要重新 `index`；若涉及 `page_blocks.jsonl` 还需要重新 `ingest`

## 你下一步可以做的增强（建议顺序）
- 加入版面分析（表格/标题/页眉页脚过滤）
- 加入 reranker（bge-reranker）提升检索精度
- 引用定位可视化（在页面图片上画出图块框）


