"""Tests for minkowski_dist."""
import pytest
from morie.fn.minkd import minkowski_dist, minkd


def test_manhattan():
    r = minkowski_dist([0, 0], [3, 4], p=1)
    assert abs(r.estimate - 7.0) < 1e-10


def test_euclidean():
    r = minkowski_dist([0, 0], [3, 4], p=2)
    assert abs(r.estimate - 5.0) < 1e-10


def test_alias():
    assert minkd is minkowski_dist


def test_bad_p():
    with pytest.raises(ValueError):
        minkowski_dist([0], [1], p=0.5)
