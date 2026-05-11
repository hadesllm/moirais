"""Launch the TIDE Go binary from within the morie Python package.

Looks for the binary in these locations (in order):
  1. ``TIDE_BIN`` environment variable
  2. Adjacent to this file (bundled in wheel)
  3. ``$PATH`` (standalone install via ``go install`` or brew)
  4. Build from source if Go toolchain is available
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_tide() -> str | None:
    if (env := os.environ.get("TIDE_BIN")) and Path(env).is_file():
        return env

    pkg_dir = Path(__file__).parent
    for name in ("tide", "tide.exe"):
        candidate = pkg_dir / name
        if candidate.is_file():
            return str(candidate)

    found = shutil.which("tide")
    if found:
        return found

    return None


def main() -> None:
    binary = _find_tide()
    if binary is None:
        print(
            "TIDE binary not found.\n"
            "Install via: go install github.com/hadesllm/morie/tide@latest\n"
            "Or set TIDE_BIN=/path/to/tide",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = subprocess.run([binary] + sys.argv[1:])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
