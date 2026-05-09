"""Test hilbert_envelope_fn."""
import numpy as np
from moirais.fn.hlbrt import hilbert_envelope_fn, alias
from moirais.fn._containers import SignalResult


class TestHilbertEnvelopeFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hilbert_envelope_fn(x)
        assert isinstance(result, SignalResult)

    def test_filtered_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hilbert_envelope_fn(x)
        assert result.filtered is not None
        assert len(result.filtered) == len(x)

    def test_n_samples(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hilbert_envelope_fn(x)
        assert result.n_samples == 256

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = hilbert_envelope_fn(x)
        assert result.name == "hilbert_envelope"

    def test_alias(self):
        assert alias is hilbert_envelope_fn
