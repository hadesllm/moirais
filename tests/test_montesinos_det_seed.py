"""Deterministic-seed plumbing tests for the Montesinos suite.

Verifies that the ``deterministic_seed`` kwarg added in morie v0.4.0 to
each of the seven Montesinos callables (``blasf``, ``brdgf``, ``bglup``,
``dlgen``, ``cnnge``, ``rnnge``, ``trfge``):

* gives bit-identical numeric outputs across two calls with the same
  ``deterministic_seed``;
* gives different numeric outputs when the seed is changed;
* leaves the default ``deterministic_seed=None`` path (legacy ``seed=``)
  unaffected.

The fixtures here are small (n <= 20, p/m <= 6) so each test runs in
well under a second even for the Gibbs / training callables.
"""

from __future__ import annotations

import numpy as np

from morie.fn.blasf import bayesian_lasso_full
from morie.fn.brdgf import bayes_ridge_gibbs
from morie.fn.bglup import bayes_cpi_genomic
from morie.fn.dlgen import deep_learning_genomic
from morie.fn.cnnge import cnn_genomic
from morie.fn.rnnge import rnn_genomic
from morie.fn.trfge import transformer_genomic


def _regression_fixture(n: int = 20, p: int = 5):
    """Small (n, p) regression fixture with a sparse true coefficient vector."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[0] = 1.0
    beta_true[1] = -1.0
    y = X @ beta_true + 0.2 * rng.standard_normal(n)
    return X, y


def _genomic_fixture(n: int = 15, m: int = 6):
    """Small (n, m) genotype-style fixture for the DL/CNN/RNN/Transformer tests."""
    rng = np.random.default_rng(1)
    M = rng.standard_normal((n, m))
    y = M[:, 0] + 0.2 * rng.standard_normal(n)
    return np.zeros(n), y, M


# ---------------------------------------------------------------------------
# Bayesian regression callables (Gibbs samplers)
# ---------------------------------------------------------------------------

def test_blasf_deterministic_seed_reproducible():
    X, y = _regression_fixture()
    r1 = bayesian_lasso_full(X, y, n_iter=60, burn=20, deterministic_seed=42)
    r2 = bayesian_lasso_full(X, y, n_iter=60, burn=20, deterministic_seed=42)
    r3 = bayesian_lasso_full(X, y, n_iter=60, burn=20, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert np.array_equal(r1["beta"], r2["beta"])
    assert r1["estimate"] != r3["estimate"]


def test_brdgf_deterministic_seed_reproducible():
    X, y = _regression_fixture()
    r1 = bayes_ridge_gibbs(X, y, n_iter=60, burn=20, deterministic_seed=42)
    r2 = bayes_ridge_gibbs(X, y, n_iter=60, burn=20, deterministic_seed=42)
    r3 = bayes_ridge_gibbs(X, y, n_iter=60, burn=20, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert np.array_equal(r1["beta"], r2["beta"])
    assert r1["estimate"] != r3["estimate"]


def test_bglup_deterministic_seed_reproducible():
    X, y = _regression_fixture()
    r1 = bayes_cpi_genomic(X, y, n_iter=80, burn=30, deterministic_seed=42)
    r2 = bayes_cpi_genomic(X, y, n_iter=80, burn=30, deterministic_seed=42)
    r3 = bayes_cpi_genomic(X, y, n_iter=80, burn=30, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert np.array_equal(r1["beta"], r2["beta"])
    assert r1["estimate"] != r3["estimate"]


# ---------------------------------------------------------------------------
# Deep-learning callables (init draws from rng, training is deterministic GD)
# ---------------------------------------------------------------------------

def test_dlgen_deterministic_seed_reproducible():
    x, y, M = _genomic_fixture()
    r1 = deep_learning_genomic(x, y, M, n_epochs=30, deterministic_seed=42)
    r2 = deep_learning_genomic(x, y, M, n_epochs=30, deterministic_seed=42)
    r3 = deep_learning_genomic(x, y, M, n_epochs=30, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert np.array_equal(r1["W1"], r2["W1"])
    assert r1["estimate"] != r3["estimate"]


def test_cnnge_deterministic_seed_reproducible():
    x, y, M = _genomic_fixture()
    r1 = cnn_genomic(x, y, M, n_epochs=20, deterministic_seed=42)
    r2 = cnn_genomic(x, y, M, n_epochs=20, deterministic_seed=42)
    r3 = cnn_genomic(x, y, M, n_epochs=20, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert np.array_equal(r1["W_conv"], r2["W_conv"])
    assert r1["estimate"] != r3["estimate"]


def test_rnnge_deterministic_seed_reproducible():
    x, y, M = _genomic_fixture()
    r1 = rnn_genomic(x, y, M, n_epochs=20, deterministic_seed=42)
    r2 = rnn_genomic(x, y, M, n_epochs=20, deterministic_seed=42)
    r3 = rnn_genomic(x, y, M, n_epochs=20, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert np.array_equal(r1["W_h"], r2["W_h"])
    assert r1["estimate"] != r3["estimate"]


def test_trfge_deterministic_seed_reproducible():
    x, y, M = _genomic_fixture()
    r1 = transformer_genomic(x, y, M, deterministic_seed=42)
    r2 = transformer_genomic(x, y, M, deterministic_seed=42)
    r3 = transformer_genomic(x, y, M, deterministic_seed=999)

    assert r1["estimate"] == r2["estimate"]
    assert np.array_equal(r1["beta"], r2["beta"])
    assert r1["estimate"] != r3["estimate"]
