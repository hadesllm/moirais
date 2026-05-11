"""Test klzkl."""
import pytest
from morie.fn.klzkl import kaluza_klein_spectrum


def test_klzkl_basic():
    r = kaluza_klein_spectrum(R=1.0, n_max=5)
    assert r.extra["masses"][1] == pytest.approx(1.0)
    assert len(r.extra["masses"]) == 6


def test_klzkl_invalid():
    with pytest.raises(ValueError):
        kaluza_klein_spectrum(R=-1.0)
