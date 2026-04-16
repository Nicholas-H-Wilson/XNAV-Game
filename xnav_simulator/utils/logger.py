# utils/logger.py — Simulation run logger for XNAV Cold Start Simulator
# XNAV Cold Start Simulator

"""
SimLogger: wraps stdlib logging and writes a JSON-lines run log to disk.

Usage
-----
    from utils.logger import get_logger, SimLogger

    log = get_logger(__name__)   # standard module-level logger
    sim_log = SimLogger()        # for structured per-run JSON logging
    sim_log.log_event("filter_init", {"n_particles": 5000, "tier": "Balanced"})
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from datetime import datetime
from typing import Any


# ── Logging configuration ─────────────────────────────────────────────────────

_LOG_DIR = pathlib.Path(__file__).parent.parent / "logs"
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger with a stream handler.

    Called once at app startup.  Subsequent module-level loggers obtained
    via get_logger(__name__) will inherit this configuration automatically.
    """
    root = logging.getLogger()
    if root.handlers:
        return   # already configured — avoid duplicate handlers on Streamlit reruns

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.  Thin wrapper for consistency."""
    return logging.getLogger(name)


# ── Structured JSON run logger ────────────────────────────────────────────────

class SimLogger:
    """Write structured JSON-lines events to logs/run_<timestamp>.jsonl.

    Each call to log_event() appends one JSON object per line to the log file.
    The log file is created lazily on the first event so sessions that never
    call log_event() leave no files on disk.
    """

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path: pathlib.Path | None = None
        self._logger = get_logger(__name__)

    @property
    def log_path(self) -> pathlib.Path:
        if self._log_path is None:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._log_path = _LOG_DIR / f"run_{self._run_id}.jsonl"
        return self._log_path

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a structured event to the run log file."""
        record = {
            "ts": time.time(),
            "run_id": self._run_id,
            "event": event_type,
            **payload,
        }
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            self._logger.warning("SimLogger: could not write log: %s", exc)

    def log_iteration(self, step: int, error_kpc: float | None, ess: float) -> None:
        """Convenience wrapper for filter iteration events."""
        self.log_event("iteration", {
            "step": step,
            "error_kpc": error_kpc,
            "ess": ess,
        })
