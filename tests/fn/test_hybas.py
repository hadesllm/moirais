"""Test hybas."""
import numpy as np
import pytest
from moirais.fn.hybas import hybas


def test_hybas_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hybas(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hybas_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hybas(flow=flow, precip=precip, n=20)
    assert r.name
