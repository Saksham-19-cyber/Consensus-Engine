"""
Trial Log Writer
=================
Streams TrialResult objects to a JSONL file as trials complete.
This enables:
  - Interruption-safe benchmark runs (partial logs are always valid)
  - Auditability — every raw trial is persisted, not just aggregated summaries
  - Direct inspection: `cat data/logs/business_deal_*.jsonl | python -m json.tool`
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models.evaluation import TrialResult
from src.config import settings

logger = logging.getLogger(__name__)


class TrialLogWriter:
    """
    Context-manager-compatible JSONL log writer for trial results.

    Usage:
        with TrialLogWriter("business_deal") as writer:
            for trial in trials:
                writer.write(trial)

    Or without context manager:
        writer = TrialLogWriter("business_deal")
        writer.open()
        writer.write(trial)
        writer.close()
    """

    def __init__(
        self,
        scenario_name: str,
        log_dir: Optional[Path] = None,
        tag: str = "",
        filepath: Optional[Path] = None,
    ):
        self.scenario_name = scenario_name
        if filepath is not None:
            self.path = Path(filepath)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_dir = log_dir or (settings.data_dir / "logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            suffix = f"_{tag}" if tag else ""
            filename = f"{scenario_name}{suffix}_{timestamp}.jsonl"
            self.path = log_dir / filename

        self._file = None
        self._count = 0

    def get_completed_keys(self) -> set[tuple[str, str, int]]:
        """Return set of (scenario_name, method, seed) tuples already in the log file."""
        keys = set()
        if not self.path.exists():
            return keys
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        keys.add((obj.get("scenario_name", ""), obj.get("method", ""), int(obj.get("seed", 0))))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Error reading completed keys from %s: %s", self.path, e)
        return keys

    def open(self):
        # Open in append mode so existing records are preserved on resume
        self._file = open(self.path, "a", encoding="utf-8")
        logger.info("trial log open (append mode): %s", self.path)
        return self

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        logger.info("trial log closed: %s (%d trials written)", self.path, self._count)

    def write(self, result: TrialResult):
        """Append a single trial result as a JSON line."""
        if self._file is None:
            raise RuntimeError("TrialLogWriter not opened. Call open() or use as context manager.")
        line = result.model_dump_json()
        self._file.write(line + "\n")
        self._file.flush()  # Flush after each write for interruption safety
        self._count += 1

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


def load_trial_log(path: Path) -> list[TrialResult]:
    """
    Load a JSONL log file back into TrialResult objects.
    Useful for re-running aggregation on previously completed runs.
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(TrialResult.model_validate(obj))
            except Exception as e:
                logger.warning("log parse error at line %d: %s", line_num, e)
    logger.info("loaded %d trials from %s", len(results), path)
    return results
