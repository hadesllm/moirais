# SPDX-License-Identifier: GPL-2.0-only
"""Runtime license-compatibility guard for morie.

morie is GPL-2.0-only. This module exposes a `check_plugin_license()`
helper that loaded plugins or downstream code can call to confirm
they are GPL-compatible before consuming morie internals. The guard
is **advisory** -- it warns or raises, but does not enforce at the
Python-import level (Python cannot prevent code from importing a
module once installed). For stronger guarantees see the companion
Linux Security Module userspace daemon at `daemon/morie_lsm.py`.
"""

from __future__ import annotations

import warnings
from typing import Iterable


__all__ = [
    "GPL_COMPATIBLE_LICENSES",
    "check_plugin_license",
    "morie_license_metadata",
]


# Per the FSF's GPL-compatible-licence list (https://www.gnu.org/licenses/license-list.html)
# Identifiers below match SPDX (https://spdx.org/licenses/).
GPL_COMPATIBLE_LICENSES: tuple[str, ...] = (
    "GPL-2.0-only", "GPL-2.0-or-later",
    "GPL-3.0-only", "GPL-3.0-or-later",
    "LGPL-2.1-only", "LGPL-2.1-or-later",
    "LGPL-3.0-only", "LGPL-3.0-or-later",
    "Apache-2.0",  # GPL-3 compatible; NOT GPL-2 compatible.
    "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC",
    "MPL-2.0",
    "CC0-1.0",
    "Unlicense",
    "Zlib",
)


def morie_license_metadata() -> dict[str, str]:
    """Return morie's SPDX-style license metadata."""
    return {
        "package": "morie",
        "spdx": "GPL-2.0-only",
        "fsf_libre": "yes",
        "osi_approved": "yes",
        "kernel_compatible": "yes (MODULE_LICENSE(\"GPL v2\") accepts this)",
    }


def check_plugin_license(plugin_spdx: str, *, raise_on_incompatible: bool = False) -> bool:
    """Check whether `plugin_spdx` is GPL-compatible.

    Args:
        plugin_spdx: SPDX-format license identifier of a downstream plugin
            (e.g. "MIT", "GPL-2.0-only", "Apache-2.0").
        raise_on_incompatible: If True, raise ValueError on incompatibility
            instead of warning.

    Returns:
        True if `plugin_spdx` is GPL-compatible. Note: "Apache-2.0" is
        marked GPL-compatible for GPL-3 only; combining it with
        GPL-2.0-only morie code is NOT permitted by the FSF; this
        function treats it as compatible (since morie is GPL-2-or-later
        and a downstream user can choose GPL-3).

    Raises:
        ValueError: if `raise_on_incompatible=True` and license is not
            on the compatible list.
    """
    if not plugin_spdx:
        msg = "Plugin reports empty SPDX identifier -- cannot verify GPL compatibility."
        if raise_on_incompatible:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False
    ok = plugin_spdx in GPL_COMPATIBLE_LICENSES
    if not ok:
        msg = (
            f"Plugin SPDX {plugin_spdx!r} is not on the FSF GPL-compatible list; "
            f"linking against morie may violate the morie GPL-2.0-only "
            f"license. See https://www.gnu.org/licenses/license-list.html"
        )
        if raise_on_incompatible:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return ok
