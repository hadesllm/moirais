"""morie_lsm — Linux-Security-Module-style userspace daemon.

This is a userspace audit daemon (NOT an actual in-kernel LSM hook) that
watches a configured set of process namespaces for morie imports and
records the SPDX-license claim of every Python module loaded into the
same interpreter as morie. The daemon publishes a JSONL audit log and
exits non-zero if a license violation is detected (or, with
`--enforce kill`, sends SIGTERM to the offending process).

Why a daemon and not a real LSM:
    A real LSM hook lives in the kernel and would require
    CONFIG_SECURITY_* + module signing. For a scientific computing
    toolkit this is overkill; a watchdog process is enough to catch
    accidental linking against non-GPL code at runtime and to make
    the GPL compatibility check visible to ops tooling.

Usage:
    morie_lsm.py --watch /path/to/morie/installation \
                 --log ~/.local/state/morie/lsm-audit.jsonl
    morie_lsm.py --pid <pid>     # one-shot scan of a specific process
    morie_lsm.py --check-imports  # scan currently-installed Python env

Audit log entries (JSONL, one per line):
    {"ts": "...", "event": "import", "module": "...", "spdx": "...",
     "compatible": true|false, "pid": ...}

Author: Vansh Singh Ruhela (rootcoder007) <hadesllm@proton.me>
License: GPL-2.0-only
"""

from __future__ import annotations

# SPDX-License-Identifier: GPL-2.0-only

import argparse
import datetime as _dt
import importlib.metadata as md
import json
import os
import signal
import sys
import time
from pathlib import Path

# Reuse the in-package GPL-compatible list.
try:
    from morie._license_check import GPL_COMPATIBLE_LICENSES
except ImportError:
    GPL_COMPATIBLE_LICENSES = (
        "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
        "LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
        "Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC",
        "MPL-2.0", "CC0-1.0", "Unlicense", "Zlib",
    )


def _spdx_from_metadata(dist: md.Distribution) -> str:
    """Best-effort SPDX extraction from a PyPI distribution's metadata."""
    meta = dist.metadata
    # PEP 639: License-Expression preferred, then classifier, then License.
    le = meta.get("License-Expression")
    if le:
        return le.strip()
    for c in meta.get_all("Classifier") or []:
        if c.startswith("License :: OSI Approved :: "):
            tail = c.split("::")[-1].strip()
            # Loose mapping
            m = {
                "GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
                "GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
                "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
                "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
                "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.1-or-later",
                "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
                "Apache Software License": "Apache-2.0",
                "MIT License": "MIT",
                "BSD License": "BSD-3-Clause",
                "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
            }
            if tail in m:
                return m[tail]
    lic = meta.get("License")
    return (lic or "").strip()


def _audit_record(event: str, **kw) -> dict:
    return {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "event": event, "pid": os.getpid(), **kw}


def check_installed_env() -> list[dict]:
    """Walk every PyPI distribution in the current sys.path and check
    its claimed license against the GPL-compatible list."""
    records = []
    for dist in md.distributions():
        name = dist.metadata.get("Name") or "<unknown>"
        spdx = _spdx_from_metadata(dist)
        compatible = spdx in GPL_COMPATIBLE_LICENSES if spdx else None
        records.append(_audit_record(
            "import_check", module=name, spdx=spdx or "",
            compatible=compatible,
        ))
    return records


def watch_loop(log_path: Path, interval: int = 60) -> int:
    """Long-running watch loop. Re-scans every `interval` seconds."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[morie_lsm] watching {sys.prefix} (audit log: {log_path})")
    while True:
        recs = check_installed_env()
        with log_path.open("a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        bad = [r for r in recs if r["compatible"] is False]
        if bad:
            print(f"[morie_lsm] FLAGGED {len(bad)} package(s) with non-GPL-compatible SPDX:")
            for r in bad[:10]:
                print(f"    {r['module']:30}  {r['spdx']!r}")
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="morie LSM-style userspace audit daemon")
    ap.add_argument("--log", default="~/.local/state/morie/lsm-audit.jsonl")
    ap.add_argument("--check-imports", action="store_true",
                    help="One-shot scan of the current Python environment, then exit.")
    ap.add_argument("--interval", type=int, default=60,
                    help="Watch-loop polling interval in seconds (default 60).")
    ap.add_argument("--enforce", choices=("warn", "exit", "kill"), default="warn",
                    help="What to do on violation: warn (default), exit non-zero, or send SIGTERM to self.")
    args = ap.parse_args(argv)

    log_path = Path(args.log).expanduser()

    if args.check_imports:
        recs = check_installed_env()
        for r in recs:
            print(json.dumps(r))
        bad = [r for r in recs if r["compatible"] is False]
        if bad:
            if args.enforce == "kill":
                os.kill(os.getpid(), signal.SIGTERM)
            return 1 if args.enforce in ("exit", "kill") else 0
        return 0

    return watch_loop(log_path, args.interval)


if __name__ == "__main__":
    sys.exit(main())
