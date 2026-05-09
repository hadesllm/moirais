"""Test dtsas."""
import numpy as np
import pytest
from moirais.fn.dtsas import dtsas


def test_dtsas_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtsas(x=x, n=50)
    assert r.value is not None


def test_dtsas_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtsas(x=x, n=50)
    assert r.name
