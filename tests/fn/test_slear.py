"""Tests for morie.fn.slear — S-learner CATE."""
import numpy as np
import pytest
from morie.fn.slear import slear


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal((n, 2))
    t = rng.binomial(1, 0.5, n).astype(float)
    y = 1.0 + 2.0 * t + 0.5 * x[:, 0] + rng.standard_normal(n) * 0.3
    return y, t, x


def test_returns_dict(data):
    r = slear(*data)
    for k in ("cate", "ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert k in r


def test_ate_close_to_truth(data):
    r = slear(*data)
    assert abs(r["ate"] - 2.0) < 0.5


def test_cate_shape(data):
    y, t, x = data
    r = slear(*data)
    assert r["cate"].shape == (len(y),)


def test_ci_brackets_ate(data):
    r = slear(*data)
    assert r["ci_lower"] <= r["ate"] <= r["ci_upper"]


def test_se_positive(data):
    r = slear(*data)
    assert r["se"] > 0


def test_method_label(data):
    r = slear(*data)
    assert r["method"] == "S-learner"


def test_logistic_model():
    rng = np.random.default_rng(1)
    n = 150
    x = rng.standard_normal((n, 2))
    t = rng.binomial(1, 0.5, n).astype(float)
    y = rng.binomial(1, 0.3 + 0.3 * t).astype(float)
    r = slear(y, t, x, model="logistic")
    assert np.isfinite(r["ate"])


def test_n_correct(data):
    y, t, x = data
    assert slear(*data)["n"] == len(y)


def test_cheatsheet():
    from morie.fn.slear import cheatsheet
    assert isinstance(cheatsheet(), str) and len(cheatsheet()) > 0
