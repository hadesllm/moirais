"""Tests for conditional mutual information."""
import numpy as np
import pytest
from moirais.fn.cmifn import conditional_mutual_information, cmifn


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    y = rng.normal(0, 1, 1000)
    z = rng.normal(0, 1, 1000)
    r = conditional_mutual_information(x, y, z, bins=8)
    assert np.isfinite(r.estimate)


def test_alias():
    assert cmifn is conditional_mutual_information


def test_length_mismatch():
    with pytest.raises(ValueError):
        conditional_mutual_information([1, 2], [1, 2, 3], [1, 2])
