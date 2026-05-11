"""Test elpd."""
import numpy as np
import pytest
from morie.fn.elpd import expected_log_pred


def test_elpd_basic():
    rng = np.random.default_rng(42)
    ll = rng.standard_normal((100, 20)) - 2.0
    r = expected_log_pred(ll)
    assert r.name == "elpd"
    assert r.value < 0


def test_elpd_shape():
    rng = np.random.default_rng(7)
    ll = rng.standard_normal((50, 10)) - 1.0
    r = expected_log_pred(ll)
    assert r.extra["S"] == 50
    assert r.extra["n"] == 10
    assert r.extra["se"] > 0
