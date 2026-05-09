"""Tests for imfex.py - IMF extraction."""
import numpy as np
from moirais.fn.imfex import imf_extract, imfex


def test_imf_extract_returns_result():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 500)
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t) + rng.standard_normal(500) * 0.1
    result = imf_extract(x, n_imfs=3)
    assert result.name == "imf_extract"
    assert "imfs" in result.extra
    assert len(result.extra["imfs"]) > 0


def test_imf_residue_exists():
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 4 * np.pi, 200)) + rng.standard_normal(200) * 0.1
    result = imf_extract(x, n_imfs=2)
    assert "residue" in result.extra
    assert len(result.extra["residue"]) == len(x)


def test_imf_alias():
    x = np.random.default_rng(42).standard_normal(200)
    result = imfex(x, n_imfs=2)
    assert result.name == "imf_extract"
