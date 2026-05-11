"""Tests for psddb.py - PSD to decibels."""
import numpy as np
import pytest
from morie.fn.psddb import psd_decibels, psddb


def test_psd_decibels_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = psd_decibels(x)
    assert result.name == "psd_decibels"
    assert "psd_db" in result.extra
    assert "freqs" in result.extra


def test_psd_decibels_array_shape():
    x = np.random.default_rng(42).standard_normal(256)
    result = psd_decibels(x)
    assert len(result.extra["psd_db"]) == len(result.extra["freqs"])


def test_psd_decibels_values_finite():
    x = np.random.default_rng(42).standard_normal(256)
    result = psd_decibels(x)
    assert np.all(np.isfinite(result.extra["psd_db"]))


def test_psddb_alias():
    x = np.random.default_rng(42).standard_normal(64)
    result = psddb(x)
    assert result.name == "psd_decibels"
