"""JSON and path helpers for reproducible experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def repo_root(start: Path | None = None) -> Path:
    """Return repository root (directory containing ``pyproject.toml``)."""
    p = start or Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[2]


def load_json(path: PathLike) -> Any:
    """Load a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: PathLike, data: Any, indent: int = 2) -> None:
    """Write ``data`` as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
