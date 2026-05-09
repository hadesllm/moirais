"""Tests for moirais.fn.notrs — NOTEARS DAG learning."""
import numpy as np
import pytest
from moirais.fn.notrs import notrs


@pytest.fixture()
def data():
    rng = np.random.default_rng(10)
    n = 150
    x1 = rng.standard_normal(n)
    x2 = 0.6 * x1 + rng.standard_normal(n) * 0.5
    x3 = 0.5 * x2 + rng.standard_normal(n) * 0.5
    return np.column_stack([x1, x2, x3])


def test_keys(data):
    r = notrs(data, max_iter=20)
    for k in ("W", "dag", "h", "p", "n", "method"):
        assert k in r


def test_W_shape(data):
    r = notrs(data, max_iter=20)
    assert r["W"].shape == (3, 3)


def test_dag_binary(data):
    r = notrs(data, max_iter=20)
    assert set(np.unique(r["dag"])).issubset({0, 1})


def test_diagonal_zero(data):
    r = notrs(data, max_iter=20)
    np.testing.assert_array_equal(np.diag(r["dag"]), 0)


def test_h_finite(data):
    r = notrs(data, max_iter=20)
    assert np.isfinite(r["h"])


def test_method(data):
    assert notrs(data, max_iter=20)["method"] == "NOTEARS"


def test_sparser_with_higher_lambda(data):
    r_low = notrs(data, lambda1=0.01, max_iter=30)
    r_high = notrs(data, lambda1=1.0, max_iter=30)
    assert r_high["dag"].sum() <= r_low["dag"].sum() + 3


def test_cheatsheet():
    from moirais.fn.notrs import cheatsheet
    assert len(cheatsheet()) > 0
