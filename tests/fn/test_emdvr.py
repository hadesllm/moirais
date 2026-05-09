"""Tests for emdvr.py - EMD variance ratio."""
import numpy as np
from moirais.fn.emdvr import emd_variance_ratio, emdvr


def test_variance_ratio_returns_result():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    imfs = [rng.standard_normal(100) for _ in range(3)]
    result = emd_variance_ratio(x, imfs)
    assert result.name == "emd_variance_ratio"
    assert "ratios" in result.extra
    assert len(result.extra["ratios"]) == 3


def test_variance_ratio_sums():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    imf1 = x * 0.6
    imf2 = x * 0.4
    result = emd_variance_ratio(x, [imf1, imf2])
    ratios = result.extra["ratios"]
    assert all(r >= 0 for r in ratios)


def test_variance_ratio_alias():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    result = emdvr(x, [x * 0.5])
    assert result.name == "emd_variance_ratio"
