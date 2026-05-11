"""Tests for morie.fn.lingm — LiNGAM causal discovery."""
import numpy as np
import pytest
from morie.fn.lingm import lingm


@pytest.fixture()
def data():
    rng = np.random.default_rng(9)
    n = 300
    e1 = rng.laplace(size=n)
    e2 = rng.laplace(size=n)
    x1 = e1
    x2 = 0.8 * x1 + e2
    return np.column_stack([x1, x2])


def test_keys(data):
    r = lingm(data)
    for k in ("B", "order", "W", "method", "p", "n"):
        assert k in r


def test_B_shape(data):
    r = lingm(data)
    assert r["B"].shape == (2, 2)


def test_order_valid(data):
    r = lingm(data)
    assert sorted(r["order"]) == [0, 1]


def test_diagonal_zero(data):
    """B should have zero diagonal (no self-causation)."""
    r = lingm(data)
    np.testing.assert_allclose(np.diag(r["B"]), 0, atol=1e-6)


def test_W_shape(data):
    r = lingm(data)
    assert r["W"].shape == (2, 2)


def test_method(data):
    assert lingm(data)["method"] == "LiNGAM"


def test_three_variables():
    rng = np.random.default_rng(42)
    n = 200
    e = rng.laplace(size=(n, 3))
    x1, x2, x3 = e[:, 0], 0.5 * e[:, 0] + e[:, 1], 0.4 * e[:, 1] + e[:, 2]
    X = np.column_stack([x1, x2, x3])
    r = lingm(X)
    assert r["p"] == 3
    assert len(r["order"]) == 3


def test_cheatsheet():
    from morie.fn.lingm import cheatsheet
    assert len(cheatsheet()) > 0
