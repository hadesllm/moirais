"""Tests for morie.fn.spflw."""
import numpy as np
from morie.fn.spflw import spflw


def test_spflw_smoke():
    rng = np.random.default_rng(42)
    result = spflw(
        origins=rng.uniform(100, 10000, size=5),
        destinations=rng.uniform(100, 10000, size=5),
        distances=rng.uniform(1, 100, size=(5, 5))
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.spflw import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
