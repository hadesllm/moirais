"""Tests for irttf — test information function."""
import numpy as np
from moirais.fn.irttf import irttf

def test_irttf_basic():
    params = {"item1": {"a": 1.0, "b": 0.0}, "item2": {"a": 1.5, "b": 1.0}}
    result = irttf(params)
    assert result is not None
    assert len(result) > 0


def test_cheatsheet():
    from moirais.fn.irttf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
