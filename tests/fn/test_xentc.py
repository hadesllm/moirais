"""Tests for cross-entropy."""
import numpy as np
import pytest
from moirais.fn.xentc import cross_entropy, xentc


def test_same_distribution():
    p = np.array([0.5, 0.5])
    r = cross_entropy(p, p)
    assert r.estimate == pytest.approx(np.log(2), abs=1e-10)


def test_positive():
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    r = cross_entropy(p, q)
    assert r.estimate > 0


def test_alias():
    assert xentc is cross_entropy


def test_zero_q_raises():
    with pytest.raises(ValueError):
        cross_entropy(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
