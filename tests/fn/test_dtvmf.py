"""Test dtvmf."""
import numpy as np
import pytest
from moirais.fn.dtvmf import dtvmf


def test_dtvmf_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtvmf(x=x, n=50)
    assert r.value is not None


def test_dtvmf_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtvmf(x=x, n=50)
    assert r.name
