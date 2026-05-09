"""Test hyrpn."""
import numpy as np
import pytest
from moirais.fn.hyrpn import hyrpn


def test_hyrpn_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyrpn(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyrpn_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyrpn(flow=flow, precip=precip, n=20)
    assert r.name
