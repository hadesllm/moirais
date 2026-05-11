"""Test gkpw."""
from morie.fn.gkpw import gkp_witten


def test_gkpw_basic():
    r = gkp_witten(phi_boundary=1.0, z_bulk=0.5, delta=3.0, d=4)
    assert r.value is not None
    assert r.name == "gkp_witten"


def test_gkpw_invalid():
    import pytest
    with pytest.raises(ValueError):
        gkp_witten(z_bulk=-1.0)
