"""Tests for moirais.fn.fgreg -- Fine-Gray competing risks regression."""

import numpy as np
import pytest

from moirais.fn.fgreg import fgreg


@pytest.fixture()
def cr_data():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    event = rng.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])
    time = rng.exponential(5, n)
    return time, event, X


def test_returns_dict(cr_data):
    time, event, X = cr_data
    r = fgreg(time, event, X, cause=1)
    assert isinstance(r, dict)
    for k in ("coefficients", "se", "subdist_hazard_ratios"):
        assert k in r


def test_coefficients_shape(cr_data):
    time, event, X = cr_data
    r = fgreg(time, event, X, cause=1)
    assert r["coefficients"].shape == (2,)


def test_shr_positive(cr_data):
    time, event, X = cr_data
    r = fgreg(time, event, X, cause=1)
    assert np.all(r["subdist_hazard_ratios"] > 0)


def test_cheatsheet():
    from moirais.fn.fgreg import cheatsheet
    assert "fine-gray" in cheatsheet().lower()
