"""Logging and progress helpers: `[TIMESTAMP] [LEVEL] [MODULE] Details...` + tqdm with rates."""
from __future__ import annotations

import logging
import os
import sys

_FMT = "[%(asctime)s] [%(levelname)s] [%(module_tag)s] %(message)s"
_configured = False


class _ModuleTagFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        record.module_tag = name.split(".")[-1] if name.startswith("gguf_weight_hacker") else name
        return True


def configure(level: str | int | None = None, logfile: str | None = None) -> None:
    """Configure toolkit logging. Safe to call repeatedly."""
    global _configured
    level = level or os.environ.get("GGUF_HACK_LOGLEVEL", "INFO")
    root = logging.getLogger("gguf_weight_hacker")
    root.setLevel(level if isinstance(level, int) else str(level).upper())
    if not _configured:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(_FMT, "%Y-%m-%dT%H:%M:%S"))
        h.addFilter(_ModuleTagFilter())
        root.addHandler(h)
        root.propagate = False
        _configured = True
    logfile = logfile or os.environ.get("GGUF_HACK_LOGFILE")
    if logfile and not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(logfile)
                           for h in root.handlers):
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter(_FMT, "%Y-%m-%dT%H:%M:%S"))
        fh.addFilter(_ModuleTagFilter())
        root.addHandler(fh)


def get_logger(module: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"gguf_weight_hacker.{module}")


def progress(iterable=None, total=None, desc="", unit="blocks", unit_scale=False, disable=None, **kw):
    """tqdm progress bar showing explicit rates (blocks/s, MB/s). Falls back to a no-op."""
    if disable is None:
        disable = os.environ.get("GGUF_HACK_NO_PROGRESS") == "1" or not sys.stderr.isatty()
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        class _Dummy:
            def __init__(self, it): self.it = it
            def __iter__(self): return iter(self.it or [])
            def update(self, n=1): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def set_postfix(self, *a, **k): pass
        return _Dummy(iterable)
    return tqdm(iterable, total=total, desc=desc, unit=unit, unit_scale=unit_scale, disable=disable,
                dynamic_ncols=True, **kw)
