#!/usr/bin/env python3
# Phase 2c addendum — scrape drids 1..79 to capture the cases the author
# discovered we'd missed (drid=70, 76, etc).

import json
import sys
import time
from pathlib import Path

PROJECT = Path("/path/to/moirais/dev/sphinx/project")
sys.path.insert(0, str(PROJECT / "libexec/config/tools/py-package"))

from moirais.siu import scrape_drid  # noqa: E402

OUT = PROJECT / "data/datasets/vsr/SIU_rows_stream_under80.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"[probe drids 1..79] → {OUT}")
rows = []
t0 = time.time()
with OUT.open("w", encoding="utf-8") as f:
    for drid in range(1, 80):
        try:
            row = scrape_drid(drid)  # auto-news-merge
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            cn = row.get("case_number") or "—"
            lang = row.get("_language", "?")
            police = (row.get("police_service") or "—")[:25]
            if cn != "—":
                print(f"  drid={drid:3}  lang={lang:7}  case={cn:14}  police={police}")
        except Exception as e:
            print(f"  drid={drid:3}  ERROR: {type(e).__name__}: {e}")

elapsed = time.time() - t0
hits = [r for r in rows if r.get("case_number")]
print(f"\n{len(hits)} of {len(rows)} drids had a case_number  ({elapsed:.1f}s)")
unique_cases = set(r["case_number"] for r in hits)
print(f"unique cases below drid=80: {len(unique_cases)}")
for c in sorted(unique_cases):
    pair = [r["drid"] for r in rows if r["case_number"] == c]
    langs = [r["_language"] for r in rows if r["case_number"] == c]
    print(f"  {c}: drids={pair} langs={langs}")
