"""Tests for chebyshev_dist."""
import pytest
from moirais.fn.chbyd import chebyshev_dist, chbyd


def test_basic():
    r = chebyshev_dist([0, 0, 0], [1, 5, 3])
    assert abs(r.estimate - 5.0) < 1e-10
    assert r.extra["argmax_dim"] == 1


def test_alias():
    assert chbyd is chebyshev_dist
