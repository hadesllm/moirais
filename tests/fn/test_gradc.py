"""Test gradient_clip."""
import numpy as np
from moirais.fn.gradc import gradient_clip, gradc
from moirais.fn._containers import DescriptiveResult


class TestGradientClip:
    def test_basic(self):
        grads = [np.array([3.0, 4.0])]
        result = gradient_clip(grads, max_norm=1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "gradient_clip"

    def test_clips_when_exceeds(self):
        grads = [np.array([3.0, 4.0])]
        result = gradient_clip(grads, max_norm=1.0)
        assert result.extra["was_clipped"] == True
        clipped_norm = np.sqrt(sum(np.sum(g**2) for g in result.extra["clipped"]))
        assert abs(clipped_norm - 1.0) < 1e-6

    def test_no_clip_when_under(self):
        grads = [np.array([0.1, 0.1])]
        result = gradient_clip(grads, max_norm=1.0)
        assert result.extra["was_clipped"] == False

    def test_alias(self):
        assert gradc is gradient_clip
