"""Tests for moirais.fn.xlear — X-learner CATE."""
import numpy as np
import pytest
from moirais.fn.xlear import xlear


@pytest.fixture()
def data():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.standard_normal((n, 2))
    t = rng.binomial(1, 0.5, n).astype(float)
    y = 1.0 + 2.0 * t + 0.3 * x[:, 0] + rng.standard_normal(n) * 0.3
    return y, t, x


def test_keys(data):
    r = xlear(*data)
    for k in ("cate", "ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert k in r


def test_ate_close_to_truth(data):
    r = xlear(*data)
    assert abs(r["ate"] - 2.0) < 0.5


def test_cate_shape(data):
    y, t, x = data
    assert xlear(*data)["cate"].shape == (len(y),)


def test_ci_valid(data):
    r = xlear(*data)
    assert r["ci_lower"] <= r["ate"] <= r["ci_upper"]


def test_method(data):
    assert xlear(*data)["method"] == "X-learner"


def test_n_correct(data):
    y, t, x = data
    assert xlear(*data)["n"] == len(y)


def test_finite(data):
    r = xlear(*data)
    assert np.isfinite(r["ate"])
    assert np.isfinite(r["se"])


def test_cheatsheet():
    from moirais.fn.xlear import cheatsheet
    assert len(cheatsheet()) > 0
