"""Test dtzpf."""
import numpy as np
import pytest
from moirais.fn.dtzpf import dtzpf


def test_dtzpf_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtzpf(x=x, n=50)
    assert r.value is not None


def test_dtzpf_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtzpf(x=x, n=50)
    assert r.name
