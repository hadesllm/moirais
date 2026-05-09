"""Tests for moirais.fn.spwgt."""
import numpy as np
from moirais.fn.spwgt import spwgt


def test_spwgt_smoke():
    rng = np.random.default_rng(42)
    result = spwgt(coords=rng.uniform(size=(20, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.spwgt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
