"""Tests for mutual_information."""
import numpy as np
import pytest
from morie.fn.mutif import mutual_information, mutif


def test_independent():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    r = mutual_information(x, y, bins=20)
    assert r.estimate < 0.2


def test_dependent():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 5000)
    y = x + rng.normal(0, 0.01, 5000)
    r = mutual_information(x, y, bins=20)
    assert r.estimate > 1.0


def test_alias():
    assert mutif is mutual_information
