"""Tests for irtif — item information function."""
import numpy as np
from morie.fn.irtif import irtif

def test_irtif_basic():
    params = {"item1": {"a": 1.0, "b": 0.0}, "item2": {"a": 1.5, "b": 1.0}}
    result = irtif(params)
    assert result is not None
    assert len(result) > 0


def test_cheatsheet():
    from morie.fn.irtif import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
