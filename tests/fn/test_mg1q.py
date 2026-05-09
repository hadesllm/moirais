"""Tests for moirais.fn.mg1q."""
import numpy as np
from moirais.fn.mg1q import mg1q


def test_mg1q_smoke():
    rng = np.random.default_rng(42)
    result = mg1q(arrival_rate=0.5, mean_service=1.0, var_service=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.mg1q import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
