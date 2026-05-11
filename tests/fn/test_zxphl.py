"""Tests for morie.fn.zxphl -- Persistence landscape"""

import numpy as np
import pytest

from morie.fn.zxphl import persistence_land


class TestPersistenceLand:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = persistence_land(data)
        assert result.value is not None

    def test_output_type(self):
        result = persistence_land(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
