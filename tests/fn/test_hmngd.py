"""Tests for moirais.fn.hmngd."""
import numpy as np
from moirais.fn.hmngd import hmngd


def test_hmngd_smoke():
    rng = np.random.default_rng(42)
    result = hmngd(a=rng.standard_normal(20), b=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.hmngd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
