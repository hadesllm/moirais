"""Tests for morie.fn.smbln."""
import numpy as np
from morie.fn.smbln import smbln


def test_smbln_smoke():
    rng = np.random.default_rng(42)
    n = 30
    X = rng.standard_normal((n, 3))
    treatment = rng.integers(0, 2, size=n).astype(float)
    result = smbln(X=X, treatment=treatment)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.smbln import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
