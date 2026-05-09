"""Test cymnf."""
import pytest
from moirais.fn.cymnf import calabi_yau_hodge


def test_cymnf_quintic():
    r = calabi_yau_hodge(h11=1, h21=101)
    assert r.value == -200.0


def test_cymnf_mirror():
    r = calabi_yau_hodge(h11=101, h21=1)
    assert r.value == 200.0


def test_cymnf_invalid():
    with pytest.raises(ValueError):
        calabi_yau_hodge(h11=-1, h21=0)
