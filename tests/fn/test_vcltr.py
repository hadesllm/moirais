"""Tests for vcltr.py - Vocal tract tube model."""
import numpy as np
from moirais.fn.vcltr import vocal_tract_model_fn, vcltr


def test_vcltr_returns_signal_result():
    areas = np.array([1.0, 2.0, 3.0, 2.5, 1.5, 1.0])
    result = vocal_tract_model_fn(areas, fs=8000.0, n_samples=512)
    assert result.name == "vocal_tract_model"
    assert result.n_samples == 512
    assert result.fs == 8000.0


def test_vcltr_impulse_starts_at_one():
    areas = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    result = vocal_tract_model_fn(areas, n_samples=256)
    assert result.filtered[0] == 1.0


def test_vcltr_reflection_coeffs_in_extra():
    areas = np.array([1.0, 2.0, 4.0])
    result = vocal_tract_model_fn(areas)
    assert "reflection_coeffs" in result.extra
    assert len(result.extra["reflection_coeffs"]) == 2


def test_vcltr_alias():
    areas = np.array([1.0, 1.5, 2.0, 1.5])
    result = vcltr(areas, n_samples=128)
    assert result.name == "vocal_tract_model"
