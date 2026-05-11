"""Tests for morie.fn.qsim."""
import numpy as np
from morie.fn.qsim import qsim


def test_qsim_smoke():
    rng = np.random.default_rng(42)
    result = qsim(arrival_rate=0.5, service_rate=0.5)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.qsim import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
