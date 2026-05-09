"""Test turns_count_fn."""
import numpy as np
from moirais.fn.turns import turns_count_fn, alias
from moirais.fn._containers import DescriptiveResult


class TestTurnsCountFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = turns_count_fn(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_non_negative(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = turns_count_fn(x)
        assert isinstance(result.value, int)
        assert result.value >= 0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = turns_count_fn(x)
        assert result.name == "turns_count"

    def test_monotone_signal_zero_turns(self):
        x = np.arange(256, dtype=float)
        result = turns_count_fn(x)
        assert result.value == 0

    def test_alias(self):
        assert alias is turns_count_fn
