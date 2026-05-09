"""Tests for lacun.py - Lacunarity analysis."""
import numpy as np
from moirais.fn.lacun import lacunarity_fn, lacun


def test_lacun_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = lacunarity_fn(x)
    assert result.name == "lacunarity"
    assert "lacunarity" in result.extra
    assert "box_sizes" in result.extra


def test_lacun_values_geq_one():
    x = np.random.default_rng(42).standard_normal(256)
    result = lacunarity_fn(x)
    assert all(v >= 1.0 for v in result.extra["lacunarity"])


def test_lacun_custom_box_sizes():
    x = np.random.default_rng(42).standard_normal(128)
    sizes = np.array([2, 4, 8, 16])
    result = lacunarity_fn(x, box_sizes=sizes)
    assert len(result.extra["lacunarity"]) == 4


def test_lacun_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = lacun(x)
    assert result.name == "lacunarity"
