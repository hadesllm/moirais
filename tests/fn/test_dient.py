"""Tests for dispersion entropy."""
import numpy as np
import pytest
from moirais.fn.dient import dispersion_entropy, dient


def test_random():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    r = dispersion_entropy(x, m=2, c=6)
    assert 0.0 <= r.estimate <= 1.0


def test_alias():
    assert dient is dispersion_entropy


def test_constant_raises():
    with pytest.raises(ValueError):
        dispersion_entropy(np.ones(100))
