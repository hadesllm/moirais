"""Tests for mutual information."""
import numpy as np
import pytest
from moirais.fn.minfo import mutual_information, minfo


def test_independent():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 10000)
    y = rng.uniform(0, 1, 10000)
    r = mutual_information(x, y, bins=10)
    assert abs(r.estimate) < 0.3


def test_correlated():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 10000)
    y = x + rng.normal(0, 0.1, 10000)
    r = mutual_information(x, y, bins=10)
    assert r.estimate > 0.5


def test_alias():
    assert minfo is mutual_information


def test_length_mismatch():
    with pytest.raises(ValueError):
        mutual_information([1, 2], [1, 2, 3])
