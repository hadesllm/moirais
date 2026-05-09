"""Test wlglm."""
import numpy as np
import pytest
from moirais.fn.wlglm import wlglm


def test_wlglm_basic():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlglm(abundance=abund, coords=coords, n=20)
    assert r.value is not None


def test_wlglm_description():
    rng = np.random.default_rng(42)
    abund = rng.poisson(10, 20)
    coords = rng.uniform(0, 100, (20, 2))
    r = wlglm(abundance=abund, coords=coords, n=20)
    assert r.name
