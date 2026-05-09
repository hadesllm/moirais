"""Test aglai."""
import numpy as np
import pytest
from moirais.fn.aglai import aglai


def test_aglai_basic():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = aglai(areas=areas, perimeters=perims, n=15)
    assert r.value is not None


def test_aglai_description():
    rng = np.random.default_rng(42)
    areas = rng.uniform(10, 500, 15)
    perims = 4 * np.sqrt(areas) * rng.uniform(0.9, 1.3, 15)
    r = aglai(areas=areas, perimeters=perims, n=15)
    assert r.name
