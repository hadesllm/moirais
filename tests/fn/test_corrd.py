"""Tests for correlation_dist."""
import numpy as np
import pytest
from morie.fn.corrd import correlation_dist, corrd


def test_perfect():
    r = correlation_dist([1, 2, 3], [2, 4, 6])
    assert abs(r.estimate) < 1e-10


def test_alias():
    assert corrd is correlation_dist


def test_too_few():
    with pytest.raises(ValueError):
        correlation_dist([1], [2])
