"""Tests for computable complexity."""
import pytest
from morie.fn.cmpls import computable_complexity, cmpls


def test_repetitive():
    r = computable_complexity("a" * 1000)
    assert r.estimate < 5.0


def test_random():
    import os
    r = computable_complexity(os.urandom(1000))
    assert r.estimate > 0


def test_alias():
    assert cmpls is computable_complexity


def test_empty_raises():
    with pytest.raises(ValueError):
        computable_complexity("")
