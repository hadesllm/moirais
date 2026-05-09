"""Test minimum_phase_correspondent."""
import numpy as np
from moirais.fn.mnphs import minimum_phase_correspondent, mnphs
from moirais.fn._containers import SignalResult


class TestMinimumPhaseCorrespondent:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = minimum_phase_correspondent(x)
        assert isinstance(result, SignalResult)

    def test_length_preserved(self):
        x = np.random.default_rng(42).standard_normal(128)
        result = minimum_phase_correspondent(x)
        assert len(result.filtered) == 128

    def test_real_output(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = minimum_phase_correspondent(x)
        assert result.filtered.dtype in (np.float64, np.float32)

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = minimum_phase_correspondent(x)
        assert result.name == "minimum_phase"

    def test_alias(self):
        assert mnphs is minimum_phase_correspondent
