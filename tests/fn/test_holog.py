"""Test holog."""
import pytest
from moirais.fn.holog import holographic_entropy


def test_holog_basic():
    r = holographic_entropy(area=4.0, G_N=1.0)
    assert r.value == pytest.approx(1.0)


def test_holog_invalid():
    with pytest.raises(ValueError):
        holographic_entropy(area=-1.0)
