"""Test matched_filter_bank (mtchb)."""
import numpy as np
from morie.fn.mtchb import matched_filter_bank, mtchb
from morie.fn._containers import DescriptiveResult


class TestMatchedFilterBank:
    def test_basic(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(200)
        templates = [rng.standard_normal(20) for _ in range(3)]
        result = matched_filter_bank(signal, templates)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "matched_filter_bank"

    def test_best_match(self):
        t = np.linspace(0, 1, 100)
        template = np.sin(2 * np.pi * 5 * t)
        signal = np.sin(2 * np.pi * 5 * t)
        templates = [np.ones(100), template, np.zeros(100)]
        result = matched_filter_bank(signal, templates)
        assert result.extra["best_template_index"] == 1

    def test_correlation_positive(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(100)
        templates = [rng.standard_normal(20)]
        result = matched_filter_bank(signal, templates)
        assert result.value >= 0

    def test_alias(self):
        assert mtchb is matched_filter_bank
