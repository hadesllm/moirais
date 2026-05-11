"""Tests for morie.fn.qrfac."""
import numpy as np
from morie.fn.qrfac import qr_factorize


def test_qrfac_smoke():
    rng = np.random.default_rng(42)
    result = qr_factorize(A=rng.standard_normal((4, 4)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.qrfac import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
