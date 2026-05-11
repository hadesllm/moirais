"""Tests for morie.fn.mtbf."""
import numpy as np
from morie.fn.mtbf import mtbf_estimate


def test_mtbf_smoke():
    rng = np.random.default_rng(42)
    result = mtbf_estimate(failure_times=np.sort(rng.uniform(0, 100, size=10)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.mtbf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
