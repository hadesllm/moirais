"""Test dtspt."""
import numpy as np
import pytest
from moirais.fn.dtspt import dtspt


def test_dtspt_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtspt(x=x, n=50)
    assert r.value is not None


def test_dtspt_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtspt(x=x, n=50)
    assert r.name
