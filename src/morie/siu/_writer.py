"""Output writers for SIU mining — CSV (the canonical 45-col table) and
JSONL (full narratives, one record per line)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from ._schema import SIU_COLUMNS


def write_csv(rows: Iterable[dict], path: Path | str,
              *, exclude_narrative: bool = True) -> int:
    """Write rows in canonical column order. Returns the count written.

    By default, `narrative_full` is excluded from the CSV (it can be
    >50KB per row and bloats the file) and written separately to JSONL
    via `write_jsonl`. Pass exclude_narrative=False to include it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = SIU_COLUMNS if not exclude_narrative else \
        [c for c in SIU_COLUMNS if c != "narrative_full"]

    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore",
                            quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in rows:
            w.writerow({k: _scalarize(row.get(k)) for k in cols})
            n += 1
    return n


def write_jsonl(rows: Iterable[dict], path: Path | str,
                *, only_keys: tuple = ("case_number", "drid", "narrative_full")) -> int:
    """Write rows as one JSON object per line. By default only keeps the
    primary key + drid + narrative_full to keep the file lean."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = {k: row.get(k) for k in only_keys}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def _scalarize(v):
    """Make value safe for CSV writer (no list/dict, no embedded newlines)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    s = str(v)
    return s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
