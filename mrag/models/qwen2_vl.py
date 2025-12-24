from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError("PyTorch not available. Install torch for Qwen2-VL.") from e


@dataclass
class Qwen2VL:
    model_name: str
    device: str = "auto"
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        _require_torch()
        import torch
        from transformers import AutoProcessor

        # transformers 版本差异：优先用专用类，否则 fallback
        try:
            from transformers import Qwen2VLForConditionalGeneration  # type: ignore

            ModelCls = Qwen2VLForConditionalGeneration
        except Exception:
            from transformers import AutoModelForVision2Seq

            ModelCls = AutoModelForVision2Seq

        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        # 经验：bfloat16 在新卡上更省显存；CPU 则用 float32
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model = ModelCls.from_pretrained(self.model_name, torch_dtype=dtype)
        self._model.to(self._device)
        self._model.eval()

    def answer_with_evidence(
        self,
        *,
        question: str,
        evidence_texts: Iterable[str],
        evidence_images: Iterable[str | Path],
    ) -> str:
        """
        用“证据约束”的方式生成答案：要求输出引用（页码/图块），避免自由发挥。
        """
        import torch

        ev_text = "\n\n".join(f"- {t}" for t in evidence_texts if t)
        img_paths = [Path(p) for p in evidence_images]

        system = (
            "你是企业文档助手。你必须只基于提供的证据回答。"
            "如果证据不足，直接说“不确定/证据不足”，并说明缺什么。"
            "回答末尾必须给出引用（例如：doc_id + page_index 或 image_id）。"
        )
        prompt = f"{system}\n\n问题：{question}\n\n证据（文本）：\n{ev_text}\n"

        images = []
        for p in img_paths[:6]:
            if p.exists():
                images.append(p)

        # Qwen2-VL processor 支持 images + text
        inputs = self._processor(text=prompt, images=images if images else None, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=False,
            )
        text = self._processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text.strip()


