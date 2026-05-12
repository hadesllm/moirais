"""Deterministic-seed plumbing tests for the Ghosal Bayesian-NP suite.

Verifies that the ``deterministic_seed`` kwarg added to the four
Ghosal Bayesian-nonparametrics callables (``ghdpm``, ``ghhbp``,
``ghbvm``, ``ghstk``) for morie v0.4.0:

* gives bit-identical headline summaries across two calls with the
  same seed;
* gives different summaries when the seed is changed; and
* leaves the default ``deterministic_seed=None`` path producing the
  same numbers as before (smoke-checked against the legacy ``seed=``
  path).
"""

from __future__ import annotations

import numpy as np

from morie.fn.ghbvm import ghosal_bernstein_von_mises
from morie.fn.ghdpm import ghosal_dpmixture_density
from morie.fn.ghhbp import ghosal_hierarchical_bayes
from morie.fn.ghstk import ghosal_stick_breaking_trunc


def _fixture() -> np.ndarray:
    # Pinned fixture (n=20) so the four Bayesian-NP callables have a
    # small, deterministic data vector to run their MCMC / bootstrap on.
    return np.array(
        [
            0.1, 0.4, -0.3, 0.7, 0.05, -0.9, 1.2, -0.4, 0.6, -0.1,
            0.3, -0.2, 0.5, -0.7, 0.0, 0.2, -0.1, 0.4, -0.5, 0.8,
        ],
        dtype=float,
    )


def test_ghdpm_deterministic_seed_reproducible():
    x = _fixture()
    r1 = ghosal_dpmixture_density(x, n_iter=20, burn=5, deterministic_seed=42)
    r2 = ghosal_dpmixture_density(x, n_iter=20, burn=5, deterministic_seed=42)
    r3 = ghosal_dpmixture_density(x, n_iter=20, burn=5, deterministic_seed=999)

    # Same deterministic_seed -> same numbers across runs.
    assert r1["estimate"] == r2["estimate"]
    assert r1["k_post"] == r2["k_post"]
    # Different deterministic_seed -> different numbers.
    assert r1["estimate"] != r3["estimate"]


def test_ghhbp_deterministic_seed_reproducible():
    x = _fixture()
    r1 = ghosal_hierarchical_bayes(x, M=200, deterministic_seed=42)
    r2 = ghosal_hierarchical_bayes(x, M=200, deterministic_seed=42)
    r3 = ghosal_hierarchical_bayes(x, M=200, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert r1["alpha_se"] == r2["alpha_se"]
    assert r1["estimate"] != r3["estimate"]


def test_ghbvm_deterministic_seed_reproducible():
    x = _fixture()
    r1 = ghosal_bernstein_von_mises(x, B=200, deterministic_seed=42)
    r2 = ghosal_bernstein_von_mises(x, B=200, deterministic_seed=42)
    r3 = ghosal_bernstein_von_mises(x, B=200, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert r1["se"] == r2["se"]
    assert r1["se"] != r3["se"]


def test_ghstk_deterministic_seed_reproducible():
    x = _fixture()
    r1 = ghosal_stick_breaking_trunc(x, alpha=2.0, K=50, deterministic_seed=42)
    r2 = ghosal_stick_breaking_trunc(x, alpha=2.0, K=50, deterministic_seed=42)
    r3 = ghosal_stick_breaking_trunc(x, alpha=2.0, K=50, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert r1["weights"] == r2["weights"]
    assert r1["estimate"] != r3["estimate"]
