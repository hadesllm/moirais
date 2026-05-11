"""Tests for arfmt.py - AR formant extraction."""
import numpy as np
from morie.fn.arfmt import ar_formant_extraction_fn, arfmt


def test_arfmt_returns_descriptive_result():
    rng = np.random.default_rng(42)
    t = np.arange(512) / 8000.0
    x = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)
    x += 0.1 * rng.standard_normal(len(x))
    result = ar_formant_extraction_fn(x, fs=8000.0)
    assert result.name == "ar_formants"
    assert "formants" in result.extra


def test_arfmt_formants_positive():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = ar_formant_extraction_fn(x, fs=8000.0, order=12)
    formants = result.extra["formants"]
    assert all(f > 0 for f in formants)


def test_arfmt_order_stored():
    x = np.random.default_rng(42).standard_normal(256)
    result = ar_formant_extraction_fn(x, order=8)
    assert result.extra["order"] == 8


def test_arfmt_alias():
    x = np.random.default_rng(42).standard_normal(256)
    result = arfmt(x, fs=4000.0, order=6)
    assert result.name == "ar_formants"
