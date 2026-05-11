#!/usr/bin/env python3
# Phase 2c — full SIU mining drid 80 → 5074.
#
# Outputs:
#   data/datasets/vsr/SIU.csv                — 65-col canonical schema
#   data/datasets/vsr/SIU_narratives.jsonl   — narrative_full bodies
#   data/datasets/vsr/SIU_summary.json       — coverage + language counts
#   data/datasets/vsr/SIU_diff_vs_1a.jsonl   — 3-way diff vs SIU1a
#   data/cache/siu/<drid>.html               — atomic html cache (already exists)
#   data/cache/siu/news_<nrid>.html          — atomic news-page cache
#
# Politeness:
#   - 250 ms between network fetches; cache hits don't sleep.
#   - tenacity retry on 5xx + transport errors (3 attempts, exp backoff).
#   - 404 sentinels at <drid>.404 — never re-fetch dead IDs.
#
# Resume:
#   - Streams rows to a JSONL "rows" file every 100 fetches so we don't
#     lose progress if the run dies mid-flight.

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path("/path/to/morie/dev/sphinx/project")
sys.path.insert(0, str(PROJECT / "libexec/config/tools/py-package"))

from morie.siu import scrape_range, write_csv, write_jsonl, SIU_COLUMNS  # noqa: E402

DRID_MIN = 80
DRID_MAX = 5074
OUT_DIR = PROJECT / "data/datasets/vsr"
ROWS_STREAM = OUT_DIR / "SIU_rows_stream.jsonl"
CSV_PATH = OUT_DIR / "SIU.csv"
JSONL_PATH = OUT_DIR / "SIU_narratives.jsonl"
SUMMARY_PATH = OUT_DIR / "SIU_summary.json"

# How often to print progress + flush stream
PROGRESS_EVERY = 50


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 2c] full SIU scrape drid {DRID_MIN}..{DRID_MAX} "
          f"({DRID_MAX - DRID_MIN + 1} IDs)")
    print(f"  rows stream → {ROWS_STREAM}")
    print(f"  CSV out     → {CSV_PATH}")
    print(f"  JSONL out   → {JSONL_PATH}")
    print(f"  expected wall-time: ~{(DRID_MAX - DRID_MIN + 1) * 0.25 / 60:.1f} min")
    print()

    rows: list[dict] = []
    languages = Counter()
    case_count = 0
    parse_failures = 0
    error_log: list[dict] = []

    t0 = time.time()
    with ROWS_STREAM.open("w", encoding="utf-8") as fout:
        for drid in range(DRID_MIN, DRID_MAX + 1):
            try:
                # Use scrape_range's gen for free politeness, but at single-drid
                # granularity so we can stream to disk per row.
                from morie.siu import scrape_drid as _sd
                row = _sd(drid)
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

                lang = row.get("_language", "unknown")
                languages[lang] += 1
                if row.get("case_number"):
                    case_count += 1

            except Exception as e:  # noqa: BLE001
                parse_failures += 1
                error_log.append({
                    "drid": drid,
                    "error": f"{type(e).__name__}: {e}",
                })

            if (drid - DRID_MIN + 1) % PROGRESS_EVERY == 0:
                fout.flush()
                pct = 100 * (drid - DRID_MIN + 1) / (DRID_MAX - DRID_MIN + 1)
                elapsed = time.time() - t0
                rate = (drid - DRID_MIN + 1) / max(elapsed, 0.001)
                eta = (DRID_MAX - drid) / max(rate, 0.001) / 60
                print(f"  drid={drid:5}  done={drid-DRID_MIN+1:5}/{DRID_MAX-DRID_MIN+1}  "
                      f"({pct:5.1f}%)  cases={case_count:5}  langs={dict(languages)}  "
                      f"err={parse_failures}  eta={eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"\n[fetch complete] {len(rows)} rows in {elapsed/60:.1f} min")

    # ── Write outputs ────────────────────────────────────────────────
    csv_n = write_csv(rows, CSV_PATH, exclude_narrative=True)
    jsonl_n = write_jsonl(
        rows, JSONL_PATH,
        only_keys=("case_number", "drid", "nrid", "narrative_full",
                   "narrative_summary", "_language"),
    )
    print(f"  SIU.csv:           {csv_n:5} rows")
    print(f"  SIU_narratives.jsonl: {jsonl_n:5} rows")

    # ── Summary ─────────────────────────────────────────────────────
    police_count = Counter(r.get("police_service") for r in rows
                           if r.get("police_service"))
    summary = {
        "drid_range": [DRID_MIN, DRID_MAX],
        "total_drids_attempted": DRID_MAX - DRID_MIN + 1,
        "rows_returned": len(rows),
        "rows_with_case_number": case_count,
        "language_counts": dict(languages),
        "police_service_counts": dict(police_count.most_common(20)),
        "parse_failures": parse_failures,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"  SIU_summary.json:  written")
    if error_log:
        (OUT_DIR / "SIU_errors.jsonl").write_text(
            "\n".join(json.dumps(e) for e in error_log)
        )
        print(f"  SIU_errors.jsonl:  {len(error_log)} errors logged")

    print(f"\n=== Phase 2c summary ===")
    print(f"  drids attempted:       {summary['total_drids_attempted']}")
    print(f"  rows with case_number: {case_count} ({100*case_count/len(rows):.1f}%)")
    print(f"  language: {dict(languages)}")
    print(f"  parse failures: {parse_failures}")
    print(f"  elapsed: {elapsed/60:.1f} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
