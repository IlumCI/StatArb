"""Append-only JSONL blotter.

Append-only on purpose: an audit trail you can rewrite is not an audit trail. Every
intended order and every fill is recorded, including the ones the risk layer blocked,
so a live session can be reconstructed afterwards.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class Blotter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        row = {"logged_at": datetime.now(UTC).isoformat(), "kind": kind, **payload}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(rows)
