"""Test clkmd."""
import numpy as np
import pytest
from moirais.fn.clkmd import clkmd


def test_clkmd_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clkmd(data=data, n=30, k=3)
    assert r.value is not None


def test_clkmd_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clkmd(data=data, n=30, k=3)
    assert r.name
