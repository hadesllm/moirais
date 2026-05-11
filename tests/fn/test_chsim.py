"""Test chsim."""
import pytest
from morie.fn.chsim import chern_simons


def test_chsim_basic():
    r = chern_simons(k=1, gauge_group="SU(2)")
    assert r.value > 0
    assert r.name == "chern_simons"


def test_chsim_invalid():
    with pytest.raises(ValueError):
        chern_simons(k=0)
