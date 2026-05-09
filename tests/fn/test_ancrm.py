"""Test anc_remove (ancrm)."""
import numpy as np
from moirais.fn.ancrm import anc_remove, ancrm
from moirais.fn._containers import SignalResult


class TestAncRemove:
    def test_basic(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(500)
        clean = np.sin(np.linspace(0, 10, 500))
        noisy = clean + noise
        result = anc_remove(noisy, noise, mu=0.01, order=8)
        assert isinstance(result, SignalResult)
        assert result.name == "anc_remove"

    def test_output_length(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(200)
        sig = rng.standard_normal(200)
        result = anc_remove(sig, noise, order=4)
        assert result.n_samples == 200

    def test_alias(self):
        assert ancrm is anc_remove
