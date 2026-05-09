"""Test raman."""
import pytest
from moirais.fn.raman import ramanujan_tau


def test_raman_tau1():
    r = ramanujan_tau(n=1)
    assert r.value == 1.0


def test_raman_tau2():
    r = ramanujan_tau(n=2)
    assert r.value == -24.0


def test_raman_invalid():
    with pytest.raises(ValueError):
        ramanujan_tau(n=0)
