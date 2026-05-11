"""Test dtcpf."""
import numpy as np
import pytest
from morie.fn.dtcpf import dtcpf


def test_dtcpf_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpf(x=x, n=50)
    assert r.value is not None


def test_dtcpf_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpf(x=x, n=50)
    assert r.name
