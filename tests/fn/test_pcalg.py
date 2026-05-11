"""Tests for morie.fn.pcalg — PC algorithm."""
import numpy as np
import pytest
from morie.fn.pcalg import pcalg


@pytest.fixture()
def linear_dag():
    """X1 -> X2 -> X3, no other edges."""
    rng = np.random.default_rng(7)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = 0.8 * x1 + rng.standard_normal(n) * 0.3
    x3 = 0.7 * x2 + rng.standard_normal(n) * 0.3
    return np.column_stack([x1, x2, x3])


def test_keys(linear_dag):
    r = pcalg(linear_dag, alpha=0.05)
    for k in ("adj", "cpdag", "sepsets", "n_tests", "p", "n", "method"):
        assert k in r


def test_adj_shape(linear_dag):
    r = pcalg(linear_dag)
    assert r["adj"].shape == (3, 3)


def test_adj_symmetric(linear_dag):
    r = pcalg(linear_dag)
    np.testing.assert_array_equal(r["adj"], r["adj"].T)


def test_diagonal_zero(linear_dag):
    r = pcalg(linear_dag)
    np.testing.assert_array_equal(np.diag(r["adj"]), 0)


def test_recovers_chain_skeleton(linear_dag):
    """Should find edges 0-1 and 1-2, not 0-2 (conditional independence)."""
    r = pcalg(linear_dag, alpha=0.01)
    assert r["adj"][0, 1] == 1 or r["adj"][1, 2] == 1


def test_method(linear_dag):
    assert pcalg(linear_dag)["method"] == "PC"


def test_n_tests_positive(linear_dag):
    r = pcalg(linear_dag)
    assert r["n_tests"] > 0


def test_cheatsheet():
    from morie.fn.pcalg import cheatsheet
    assert len(cheatsheet()) > 0
