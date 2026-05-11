"""Tests for morie.fn.fmea."""
import numpy as np
from morie.fn.fmea import fmea


def test_fmea_smoke():
    rng = np.random.default_rng(42)
    result = fmea(
        severity=rng.integers(1, 11, size=10),
        occurrence=rng.integers(1, 11, size=10),
        detection=rng.integers(1, 11, size=10)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.fmea import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
