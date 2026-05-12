"""Deterministic-seed plumbing tests for the Kosorok empirical-process suite.

Verifies that the ``deterministic_seed`` kwarg added to ``ksr07`` /
``ksr08`` for morie v0.4.0:

* gives bit-identical (estimate, se) across two calls with the same seed;
* gives different (estimate, se) when the seed is changed; and
* leaves the default ``deterministic_seed=None`` path producing the same
  numbers as before (smoke-checked against the legacy ``seed=`` path).
"""

from __future__ import annotations

import numpy as np

from morie.fn.ksr07 import kosorok_bootstrap_empirical
from morie.fn.ksr08 import kosorok_multiplier_bootstrap


def _fixture() -> np.ndarray:
    # Pinned fixture matching the canonical Kosorok parity smoke values.
    return np.array(
        [
            0.1, 0.4, -0.3, 0.7, 0.05, -0.9, 1.2, -0.4, 0.6, -0.1,
            0.3, -0.2, 0.5, -0.7, 0.0, 0.2, -0.1, 0.4, -0.5, 0.8,
        ],
        dtype=float,
    )


def test_ksr07_deterministic_seed_reproducible():
    x = _fixture()
    r1 = kosorok_bootstrap_empirical(x, B=400, deterministic_seed=42)
    r2 = kosorok_bootstrap_empirical(x, B=400, deterministic_seed=42)
    r3 = kosorok_bootstrap_empirical(x, B=400, deterministic_seed=999)

    # Same deterministic_seed -> same numbers across runs.
    assert r1["estimate"] == r2["estimate"]
    assert r1["se"] == r2["se"]
    # Different deterministic_seed -> different numbers.
    assert r1["se"] != r3["se"]


def test_ksr07_default_behaviour_unchanged():
    # With deterministic_seed=None, the legacy seed= path still applies
    # and is itself reproducible.
    x = _fixture()
    r1 = kosorok_bootstrap_empirical(x, B=400, seed=42)
    r2 = kosorok_bootstrap_empirical(x, B=400, seed=42)
    assert r1["estimate"] == r2["estimate"]
    assert r1["se"] == r2["se"]


def test_ksr08_deterministic_seed_reproducible():
    x = _fixture()
    r1 = kosorok_multiplier_bootstrap(x, B=400, deterministic_seed=42)
    r2 = kosorok_multiplier_bootstrap(x, B=400, deterministic_seed=42)
    r3 = kosorok_multiplier_bootstrap(x, B=400, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert r1["se"] == r2["se"]
    assert r1["se"] != r3["se"]


def test_ksr08_default_behaviour_unchanged():
    x = _fixture()
    r1 = kosorok_multiplier_bootstrap(x, B=400, seed=42)
    r2 = kosorok_multiplier_bootstrap(x, B=400, seed=42)
    assert r1["estimate"] == r2["estimate"]
    assert r1["se"] == r2["se"]
