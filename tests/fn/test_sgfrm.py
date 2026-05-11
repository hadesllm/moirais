"""Test frame_signal (sgfrm)."""
import numpy as np
from morie.fn.sgfrm import frame_signal, sgfrm
from morie.fn._containers import DescriptiveResult


class TestFrameSignal:
    def test_basic(self):
        x = np.arange(100, dtype=float)
        result = frame_signal(x, frame_len=20, hop_len=10)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "frame_signal"

    def test_frame_count(self):
        x = np.arange(100, dtype=float)
        result = frame_signal(x, frame_len=20, hop_len=10)
        assert result.value == 9.0
        assert result.extra["frames"].shape == (9, 20)

    def test_alias(self):
        assert sgfrm is frame_signal
