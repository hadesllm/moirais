"""Tests for morie.fn.lifeq."""
import numpy as np
from morie.fn.lifeq import lifeq


def test_lifeq_smoke():
    deaths = np.array([50, 30, 20, 40, 60], dtype=float)
    population = np.array([10000, 9000, 8000, 7000, 6000], dtype=float)
    result = lifeq(deaths=deaths, population=population)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.lifeq import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
