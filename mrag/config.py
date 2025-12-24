from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MRAGConfig:
    raw: dict[str, Any]

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["paths"]["data_dir"])


def load_config(path: str | Path) -> MRAGConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("config file must be a YAML mapping")
    return MRAGConfig(raw=raw)


