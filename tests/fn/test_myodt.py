"""Test myogram_onset (myodt)."""
import numpy as np
from morie.fn.myodt import myogram_onset, myodt
from morie.fn._containers import DescriptiveResult


class TestMyogramOnset:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(500) * 0.01,
                            rng.standard_normal(200) * 2.0,
                            rng.standard_normal(300) * 0.01])
        result = myogram_onset(x, fs=1000.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "myogram_onset"

    def test_detects_onset(self):
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(500) * 0.01,
                            rng.standard_normal(200) * 5.0,
                            rng.standard_normal(300) * 0.01])
        result = myogram_onset(x, fs=1000.0, threshold_factor=3.0)
        assert result.value >= 1

    def test_rms_computed(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        result = myogram_onset(x, fs=1000.0)
        assert len(result.extra["rms"]) == len(x)

    def test_alias(self):
        assert myodt is myogram_onset
