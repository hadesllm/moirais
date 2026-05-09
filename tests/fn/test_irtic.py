"""Tests for irtic — item characteristic curve."""
import numpy as np
from moirais.fn.irtic import irtic

def test_irtic_basic():
    params = {"item1": {"a": 1.0, "b": 0.0}}
    result = irtic(params)
    assert result is not None
    assert len(result) > 0


def test_cheatsheet():
    from moirais.fn.irtic import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
