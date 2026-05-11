"""Test regge."""
import pytest
from morie.fn.regge import regge_trajectory


def test_regge_basic():
    r = regge_trajectory(alpha0=0.5, alpha_prime=0.9, s=1.0)
    assert r.value == pytest.approx(1.4)


def test_regge_name():
    r = regge_trajectory()
    assert r.name == "regge_trajectory"
