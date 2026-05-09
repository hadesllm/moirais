"""Tests for moirais.fn.rci."""
import numpy as np
from moirais.fn.rci import rci


def test_rci_smoke():
    rng = np.random.default_rng(42)
    result = rci(
        pre=rng.standard_normal(20),
        post=rng.integers(0, 2, size=20).astype(float),
        se_meas=2.5
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.rci import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
