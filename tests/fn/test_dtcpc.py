"""Test dtcpc."""
import numpy as np
import pytest
from morie.fn.dtcpc import dtcpc


def test_dtcpc_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpc(x=x, n=50)
    assert r.value is not None


def test_dtcpc_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpc(x=x, n=50)
    assert r.name
