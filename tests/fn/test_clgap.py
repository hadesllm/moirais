"""Test clgap."""
import numpy as np
import pytest
from morie.fn.clgap import clgap


def test_clgap_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clgap(data=data, n=30, k=3)
    assert r.value is not None


def test_clgap_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clgap(data=data, n=30, k=3)
    assert r.name
