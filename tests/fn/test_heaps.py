"""Tests for Heaps' law."""
import pytest
from morie.fn.heaps import heaps_law, heaps


def test_basic():
    tokens = ["the", "cat", "sat", "on", "the", "mat"] * 100
    r = heaps_law(tokens)
    assert 0 < r.estimate < 1.5


def test_alias():
    assert heaps is heaps_law


def test_too_few():
    with pytest.raises(ValueError):
        heaps_law(["one"])
