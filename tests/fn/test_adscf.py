"""Test adscf."""
import pytest
from moirais.fn.adscf import ads_cft_dictionary


def test_adscf_from_delta():
    r = ads_cft_dictionary(delta=3.0, d=4)
    assert r.extra["m_squared"] == pytest.approx(-3.0)


def test_adscf_from_mass():
    r = ads_cft_dictionary(m_squared=0.0, d=4)
    assert r.value == pytest.approx(4.0)


def test_adscf_bf_violation():
    with pytest.raises(ValueError):
        ads_cft_dictionary(m_squared=-10.0, d=4)
