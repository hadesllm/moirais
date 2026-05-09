"""Tests for irtab — ability estimation."""
import numpy as np
from moirais.fn.irtab import irtab

def test_irtab_basic():
    responses = np.array([1, 0, 1, 1, 0])
    params = {f"item{i}": {"a": 1.0, "b": float(i-2)*0.5} for i in range(5)}
    result = irtab(responses, params)
    assert result is not None


def test_cheatsheet():
    from moirais.fn.irtab import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
