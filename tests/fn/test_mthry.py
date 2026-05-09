"""Test mthry."""
import pytest
from moirais.fn.mthry import m_theory_dimension


def test_mthry_basic():
    r = m_theory_dimension(g_s=0.1, l_s=1.0)
    assert r.value == 11.0
    assert r.extra["R_11"] == pytest.approx(0.1)


def test_mthry_invalid():
    with pytest.raises(ValueError):
        m_theory_dimension(g_s=0.0)
