"""Tests for moirais.fn.mmcq."""
import numpy as np
from moirais.fn.mmcq import mmcq


def test_mmcq_smoke():
    rng = np.random.default_rng(42)
    result = mmcq(arrival_rate=0.5, service_rate=0.5)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.mmcq import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
