"""Test lr_schedule."""
import numpy as np
from morie.fn.lrsch import lr_schedule, lrsch
from morie.fn._containers import DescriptiveResult


class TestLrSchedule:
    def test_basic(self):
        result = lr_schedule(500, warmup=100, total=1000)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "lr_schedule"

    def test_warmup(self):
        result = lr_schedule(0, warmup=100, total=1000, base_lr=1e-3)
        assert result.value == 0.0

    def test_warmup_midpoint(self):
        result = lr_schedule(50, warmup=100, total=1000, base_lr=1e-3)
        assert abs(result.value - 5e-4) < 1e-8

    def test_cosine_end(self):
        result = lr_schedule(1000, warmup=100, total=1000, base_lr=1e-3, min_lr=1e-5)
        assert result.value <= 1e-3

    def test_linear(self):
        result = lr_schedule(550, warmup=100, total=1000, type="linear", base_lr=1e-3, min_lr=0.0)
        assert 0.0 <= result.value <= 1e-3

    def test_constant(self):
        result = lr_schedule(500, warmup=0, total=1000, type="constant", base_lr=3e-4)
        assert abs(result.value - 3e-4) < 1e-10

    def test_alias(self):
        assert lrsch is lr_schedule
