"""Tests for morie.fn.ratch -- data repair pipeline."""

import numpy as np
import pandas as pd
from morie.fn.ratch import repair_pipeline, ratch
from morie.fn._containers import DescriptiveResult


class TestRatch:
    def test_alias(self):
        assert ratch is repair_pipeline

    def test_fills_nan(self):
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        r = repair_pipeline(arr)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n_filled"] == 1
        assert not np.any(np.isnan(r.value))

    def test_clips_outliers(self):
        arr = np.concatenate([np.ones(50), [100.0]])
        r = repair_pipeline(arr, clip_sigma=2.0)
        assert r.extra["n_clipped"] >= 1
