"""Tests for cross_entropy."""
import numpy as np
import pytest
from moirais.fn.xent import cross_entropy, xent


def test_same_dist():
    p = [0.25, 0.25, 0.25, 0.25]
    r = cross_entropy(p, p)
    h = -4 * 0.25 * np.log(0.25)
    assert abs(r.estimate - h) < 1e-10


def test_alias():
    assert xent is cross_entropy


def test_length_mismatch():
    with pytest.raises(ValueError):
        cross_entropy([0.5, 0.5], [1])


def test_zero_q():
    with pytest.raises(ValueError):
        cross_entropy([0.5, 0.5], [1, 0])
