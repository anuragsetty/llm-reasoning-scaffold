#!/usr/bin/env python3
"""
Validate MAWPS-style JSON in ``data/raw`` and print schema expectations.

This repository ships MAWPS train/test splits under ``data/raw`` for offline
reproduction. Use this script to confirm files exist before running experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/foo.py`` without installation
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from utils.io import load_json, repo_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MAWPS JSON artifacts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root(),
        help="Repository root (defaults to auto-detected root).",
    )
    args = parser.parse_args()
    root: Path = args.root

    train = root / "data" / "raw" / "mawps_train.json"
    test = root / "data" / "raw" / "mawps_test.json"
    missing = [p for p in (train, test) if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(1)

    tr = load_json(train)
    te = load_json(test)
    assert isinstance(tr, list) and isinstance(te, list), "Expected JSON arrays"
    for label, rows in ("train", tr), ("test", te):
        for i, row in enumerate(rows[:1]):
            assert "problem" in row and "equation" in row, f"{label} row {i} missing keys"
    print(f"OK: {train.name} ({len(tr)} rows), {test.name} ({len(te)} rows).")


if __name__ == "__main__":
    main()
