"""Atomic file writes shared by the data pipeline, trainer and experiment runner.

Every writer goes through a temporary file in the destination directory and `os.replace`.
On Windows, `os.replace` fails with PermissionError (WinError 5/32) while another process briefly
holds the destination open (antivirus, indexer, a reader); v2 lost `selectors.json` updates this
way. The final replace is therefore retried with a short backoff before giving up.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Iterator

RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)  # about 6 s in total


def replace_with_retry(src: str | os.PathLike, dst: str | os.PathLike, delays=RETRY_DELAYS) -> None:
    """`os.replace(src, dst)`, retried on PermissionError (Windows file locks)."""
    for wait in (*delays, None):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if wait is None:
                raise
            time.sleep(wait)


@contextlib.contextmanager
def atomic_path(path: str | os.PathLike) -> Iterator[Path]:
    """Yield a temporary path next to `path`; on success it replaces `path` atomically.

        with atomic_path(out) as tmp:
            torch.save(obj, tmp)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        yield tmp
        replace_with_retry(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def atomic_write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding=encoding)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)
