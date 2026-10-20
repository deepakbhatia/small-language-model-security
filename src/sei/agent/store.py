"""Filesystem JSON case store (pilot-scale; swap for DB later)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from sei.agent.types import CaseRecord

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE_DIR = REPO_ROOT / "data" / "cases"


class CaseStore:
    def __init__(self, root: Path | None = None):
        env = os.environ.get("SEI_CASE_DIR")
        self.root = Path(env) if env else (root or DEFAULT_CASE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, case_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)[:120]
        return self.root / f"{safe}.json"

    def save(self, case: CaseRecord) -> None:
        path = self._path(case.case_id)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(case.to_dict(), indent=2) + "\n"
        with self._lock:
            tmp.write_text(payload)
            tmp.replace(path)

    def get(self, case_id: str) -> CaseRecord | None:
        path = self._path(case_id)
        if not path.exists():
            return None
        return CaseRecord.from_dict(json.loads(path.read_text()))

    def list_ids(self, *, limit: int = 100) -> list[str]:
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files[:limit]]
