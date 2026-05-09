"""Test sawpl."""
import numpy as np
import pytest
from moirais.fn.sawpl import sawpl


def test_sawpl_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawpl(values=vals, n=25)
    assert r.value is not None


def test_sawpl_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawpl(values=vals, n=25)
    assert r.name
