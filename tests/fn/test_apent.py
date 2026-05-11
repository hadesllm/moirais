"""Tests for approximate entropy."""
import numpy as np
import pytest
from morie.fn.apent import approximate_entropy, apent


def test_regular():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    r = approximate_entropy(x, m=2)
    assert r.estimate >= 0


def test_alias():
    assert apent is approximate_entropy


def test_too_short():
    with pytest.raises(ValueError):
        approximate_entropy([1, 2])
