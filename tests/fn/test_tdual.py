"""Test tdual."""
import pytest
from morie.fn.tdual import t_duality


def test_tdual_basic():
    r = t_duality(R=2.0, alpha_prime=1.0)
    assert r.value == pytest.approx(0.5)


def test_tdual_self_dual():
    r = t_duality(R=1.0, alpha_prime=1.0)
    assert r.extra["is_self_dual"]


def test_tdual_invalid():
    with pytest.raises(ValueError):
        t_duality(R=-1.0)
