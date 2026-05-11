"""Tests for Cramer-Rao bound."""
import pytest
from morie.fn.cramr import cramer_rao_bound, cramr


def test_basic():
    r = cramer_rao_bound(fisher_info=4.0, n=100)
    assert r.estimate == pytest.approx(1.0 / 400, rel=1e-10)


def test_alias():
    assert cramr is cramer_rao_bound


def test_negative_fi_raises():
    with pytest.raises(ValueError):
        cramer_rao_bound(fisher_info=-1.0)
