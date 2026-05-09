"""Test sawcn."""
import numpy as np
import pytest
from moirais.fn.sawcn import sawcn


def test_sawcn_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawcn(values=vals, n=25)
    assert r.value is not None


def test_sawcn_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawcn(values=vals, n=25)
    assert r.name
