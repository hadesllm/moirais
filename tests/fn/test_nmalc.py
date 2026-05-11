"""Tests for morie.fn.nmalc -- Alpha-NOMINATE convergence"""

import numpy as np
import pytest

from morie.fn.nmalc import alpha_nom_conv


class TestAlphaNomConv:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = alpha_nom_conv(data)
        assert result.value is not None

    def test_output_type(self):
        result = alpha_nom_conv(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
