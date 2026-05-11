"""Tests for morie.fn.svbnz -- Banzhaf power index spatial"""

import numpy as np
import pytest

from morie.fn.svbnz import banzhaf_spatial


class TestBanzhafSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = banzhaf_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = banzhaf_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
