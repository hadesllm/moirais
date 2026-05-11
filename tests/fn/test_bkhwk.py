"""Test bkhwk."""
import pytest
from morie.fn.bkhwk import bekenstein_hawking


def test_bkhwk_basic():
    r = bekenstein_hawking(M=1.0)
    assert r.value > 0
    assert r.extra["hawking_temperature"] > 0


def test_bkhwk_invalid():
    with pytest.raises(ValueError):
        bekenstein_hawking(M=0.0)
