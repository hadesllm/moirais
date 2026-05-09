"""Test dbran."""
import pytest
import numpy as np
from moirais.fn.dbran import d_brane_tension


def test_dbran_d3():
    r = d_brane_tension(p=3, g_s=1.0, alpha_prime=1.0)
    expected = 1.0 / ((2 * np.pi) ** 3)
    assert r.value == pytest.approx(expected)


def test_dbran_invalid():
    with pytest.raises(ValueError):
        d_brane_tension(p=10)
