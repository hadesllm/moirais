"""Test dtlkj."""
import numpy as np
import pytest
from moirais.fn.dtlkj import dtlkj


def test_dtlkj_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtlkj(x=x, n=50)
    assert r.value is not None


def test_dtlkj_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtlkj(x=x, n=50)
    assert r.name
