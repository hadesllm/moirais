"""Test orbfd."""
import pytest
from morie.fn.orbfd import orbifold_spectrum


def test_orbfd_z3():
    r = orbifold_spectrum(group_order=3, d=6)
    assert r.value == 27.0


def test_orbfd_invalid():
    with pytest.raises(ValueError):
        orbifold_spectrum(group_order=1)
