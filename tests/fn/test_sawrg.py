"""Test sawrg."""
import numpy as np
import pytest
from moirais.fn.sawrg import sawrg


def test_sawrg_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawrg(values=vals, n=25)
    assert r.value is not None


def test_sawrg_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawrg(values=vals, n=25)
    assert r.name
