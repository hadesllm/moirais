"""Tests for morie.fn.stdyn — Dynamic spatio-temporal state-space model."""

import numpy as np
import pytest

from morie.fn.stdyn import stdyn


class TestStdyn:

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3))
        result = stdyn(data)
        assert result["filtered_states"].shape == (10, 3)
        assert result["log_likelihood"] != 0.0

    def test_custom_transition(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 4))
        G = 0.9 * np.eye(4)
        result = stdyn(data, transition=G)
        assert result["T"] == 8
        assert result["n"] == 4

    def test_transition_shape_raises(self):
        with pytest.raises(ValueError, match="transition must be"):
            stdyn(np.ones((5, 3)), transition=np.eye(4))
