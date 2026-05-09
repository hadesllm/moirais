"""Test dtbvn."""
import numpy as np
import pytest
from moirais.fn.dtbvn import dtbvn


def test_dtbvn_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbvn(x=x, n=50)
    assert r.value is not None


def test_dtbvn_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbvn(x=x, n=50)
    assert r.name
