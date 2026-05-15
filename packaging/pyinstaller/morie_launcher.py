"""Entry point frozen into the standalone morie bundle by PyInstaller.

This script is what PyInstaller turns into the bundled `morie` executable.
It hands straight off to the normal CLI, so the bundled command behaves
exactly like `python -m morie` or the pip-installed `morie` console script.
"""

import sys

from morie.runner import main

if __name__ == "__main__":
    sys.exit(main())
