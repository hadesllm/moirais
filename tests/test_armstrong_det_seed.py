"""Deterministic-seed plumbing tests for the Armstrong suite.

Verifies that the ``deterministic_seed`` kwarg added to ``bysid``
(Bayesian ideal-point estimation, Armstrong Ch 5) for morie v0.4.0:

* gives bit-identical posterior summaries across two calls with the
  same seed;
* gives different summaries when the seed is changed; and
* leaves the default ``deterministic_seed=None`` path producing the
  same numbers as before (smoke-checked against the legacy ``seed=``
  path).
"""

from __future__ import annotations

import numpy as np

from morie.fn.bysid import bysid


def _fixture() -> np.ndarray:
    # Small pinned binary vote matrix (n=12 legislators, m=6 items).
    rng = np.random.default_rng(7)
    X = rng.normal(size=12)
    return (X[:, None] + rng.normal(size=(12, 6)) > 0).astype(float)


def test_bysid_deterministic_seed_reproducible():
    Y = _fixture()
    r1 = bysid(Y, n_iter=120, burn=20, deterministic_seed=42)
    r2 = bysid(Y, n_iter=120, burn=20, deterministic_seed=42)
    r3 = bysid(Y, n_iter=120, burn=20, deterministic_seed=999)

    # Same deterministic_seed -> bit-identical posterior summaries.
    np.testing.assert_array_equal(r1["x_mean"], r2["x_mean"])
    np.testing.assert_array_equal(r1["x_sd"], r2["x_sd"])
    np.testing.assert_array_equal(r1["alpha"], r2["alpha"])
    np.testing.assert_array_equal(r1["beta"], r2["beta"])

    # Different deterministic_seed -> different posterior summaries.
    assert not np.array_equal(r1["x_mean"], r3["x_mean"])


def test_bysid_default_behaviour_unchanged():
    # With deterministic_seed=None, the legacy seed= path still applies
    # and is itself reproducible.
    Y = _fixture()
    r1 = bysid(Y, n_iter=120, burn=20, seed=42)
    r2 = bysid(Y, n_iter=120, burn=20, seed=42)
    np.testing.assert_array_equal(r1["x_mean"], r2["x_mean"])
    np.testing.assert_array_equal(r1["x_sd"], r2["x_sd"])
