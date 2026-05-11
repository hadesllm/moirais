"""Test dtcpg2."""
import numpy as np
import pytest
from morie.fn.dtcpg2 import dtcpg2


def test_dtcpg2_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpg2(x=x, n=50)
    assert r.value is not None


def test_dtcpg2_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpg2(x=x, n=50)
    assert r.name
