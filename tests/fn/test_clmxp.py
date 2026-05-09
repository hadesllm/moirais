"""Test clmxp."""
import numpy as np
import pytest
from moirais.fn.clmxp import clmxp


def test_clmxp_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clmxp(data=data, n=30, k=3)
    assert r.value is not None


def test_clmxp_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clmxp(data=data, n=30, k=3)
    assert r.name
