"""Test homomorphic_filter (hmflt)."""
import numpy as np
import pytest

from moirais.fn.hmflt import homomorphic_filter, hmflt
from moirais.fn._containers import SignalResult


class TestHomomorphicFilter:
    def test_basic(self):
        x = np.abs(np.random.default_rng(42).standard_normal(256)) + 1.0
        result = homomorphic_filter(x)
        assert isinstance(result, SignalResult)
        assert result.name == "homomorphic_filter"

    def test_output_shape(self):
        x = np.abs(np.random.default_rng(42).standard_normal(256)) + 1.0
        result = homomorphic_filter(x)
        assert result.n_samples == 256
        assert result.filtered is not None
        assert len(result.filtered) == 256

    def test_fs_stored(self):
        x = np.abs(np.random.default_rng(42).standard_normal(128)) + 1.0
        result = homomorphic_filter(x, fs=44100.0)
        assert result.fs == 44100.0

    def test_alias(self):
        assert hmflt is homomorphic_filter
