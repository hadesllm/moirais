"""Test witin."""
import pytest
from morie.fn.witin import witten_index


def test_witin_basic():
    r = witten_index(beta=1.0)
    assert r.value == pytest.approx(0.0)
    assert r.extra["susy_broken"]


def test_witin_invalid():
    with pytest.raises(ValueError):
        witten_index(beta=0.0)
