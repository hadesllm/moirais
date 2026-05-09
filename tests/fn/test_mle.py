"""Tests for mle_gaussian."""
import numpy as np
import pytest
from moirais.fn.mle import mle_gaussian, mle


def test_known():
    x = [2, 4, 6, 8, 10]
    r = mle_gaussian(x)
    assert abs(r.estimate - 6.0) < 1e-10
    assert abs(r.extra["sigma2"] - 8.0) < 1e-10


def test_alias():
    assert mle is mle_gaussian


def test_too_few():
    with pytest.raises(ValueError):
        mle_gaussian([1])


def test_large():
    rng = np.random.default_rng(42)
    x = rng.normal(5, 2, 10000)
    r = mle_gaussian(x)
    assert abs(r.estimate - 5.0) < 0.1
