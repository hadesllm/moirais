"""Tests for Kolmogorov complexity."""
import pytest
from moirais.fn.kolcm import kolmogorov_complexity, kolcm


def test_random_high():
    import os
    data = os.urandom(1000)
    r = kolmogorov_complexity(data)
    assert r.estimate > 0.5


def test_repetitive_low():
    data = b"a" * 1000
    r = kolmogorov_complexity(data)
    assert r.estimate < 0.1


def test_string():
    r = kolmogorov_complexity("hello world")
    assert r.estimate > 0


def test_alias():
    assert kolcm is kolmogorov_complexity


def test_empty_raises():
    with pytest.raises(ValueError):
        kolmogorov_complexity(b"")
