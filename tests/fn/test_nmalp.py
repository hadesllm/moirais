"""Tests for morie.fn.nmalp -- Alpha-NOMINATE posterior"""

import numpy as np
import pytest

from morie.fn.nmalp import alpha_nom_post


class TestAlphaNomPost:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = alpha_nom_post(data)
        assert result.value is not None

    def test_output_type(self):
        result = alpha_nom_post(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
