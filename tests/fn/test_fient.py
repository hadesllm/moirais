"""Tests for Fisher information."""
import numpy as np
import pytest
from moirais.fn.fient import fisher_information, fient


def test_positive():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    r = fisher_information(x, bins=50)
    assert r.estimate > 0


def test_alias():
    assert fient is fisher_information


def test_too_few():
    with pytest.raises(ValueError):
        fisher_information([1, 2])
