"""Tests for morie.fn.isisz — inter-spike interval analysis."""
import numpy as np
import pytest

from morie.fn.isisz import isi_analyze, isisz


def test_regular_spikes():
    spikes = np.arange(0, 1, 0.1)
    result = isi_analyze(spikes)
    assert abs(result.value - 0.1) < 1e-10
    assert result.extra["cv"] < 1e-10


def test_cv_positive_for_irregular():
    rng = np.random.default_rng(42)
    spikes = np.sort(rng.uniform(0, 10, 50))
    result = isi_analyze(spikes)
    assert result.extra["cv"] > 0


def test_too_few_raises():
    with pytest.raises(ValueError):
        isi_analyze([1.0])


def test_alias():
    assert isisz is isi_analyze
