"""Test hyspi."""
import numpy as np
import pytest
from morie.fn.hyspi import hyspi


def test_hyspi_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyspi(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyspi_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyspi(flow=flow, precip=precip, n=20)
    assert r.name
