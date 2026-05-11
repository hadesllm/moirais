"""Tests for earth_mover_dist."""
import pytest
from morie.fn.emdd import earth_mover_dist, emdd


def test_same():
    r = earth_mover_dist([0.5, 0.5], [0.5, 0.5])
    assert abs(r.estimate) < 1e-10


def test_shifted():
    r = earth_mover_dist([1, 0, 0], [0, 0, 1])
    assert r.estimate > 0


def test_alias():
    assert emdd is earth_mover_dist


def test_length_mismatch():
    with pytest.raises(ValueError):
        earth_mover_dist([1], [1, 2])
