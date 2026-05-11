"""Test dtcpg."""
import numpy as np
import pytest
from morie.fn.dtcpg import dtcpg


def test_dtcpg_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpg(x=x, n=50)
    assert r.value is not None


def test_dtcpg_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpg(x=x, n=50)
    assert r.name
