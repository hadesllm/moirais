"""Test matched_filter_detect (mtchf)."""
import numpy as np
import pytest

from moirais.fn.mtchf import matched_filter_detect, mtchf
from moirais.fn._containers import SignalResult


class TestMatchedFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = matched_filter_detect(x, template=x[:20])
        assert isinstance(result, SignalResult)
        assert result.name == "matched_filter_detect"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = matched_filter_detect(x, template=x[:20])
        assert result.n_samples == 256

    def test_filtered_not_none(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = matched_filter_detect(x, template=x[:20])
        assert result.filtered is not None

    def test_different_template(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        template = rng.standard_normal(30)
        result = matched_filter_detect(x, template=template)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert mtchf is matched_filter_detect
