"""Tests for 'rv_piecelin.py' file
"""
import numpy as np
from numpy.testing import assert_array_equal

from randomvars.rv_piecelin import rv_piecelin


class TestRVPiecelin:
    """Tests for `rv_piecelin` class
    """

    def test_init(self):
        """Test correct initialization
        """
        x_ref = np.array([0, 1, 2])
        y_ref = np.array([0, 1, 0])

        # Simple case with numpy input
        rv_1 = rv_piecelin(x=x_ref, y=y_ref)
        x_1_out, y_1_out = rv_1.get_grid()
        assert_array_equal(x_1_out, x_ref)
        assert_array_equal(y_1_out, y_ref)

        # Simple case with non-numpy input
        rv_2 = rv_piecelin(x=x_ref.tolist(), y=y_ref.tolist())
        x_2_out, y_2_out = rv_2.get_grid()
        assert_array_equal(x_2_out, x_ref)
        assert_array_equal(y_2_out, y_ref)

        # Check if `y` is normalized
        rv_3 = rv_piecelin(x=x_ref, y=10 * y_ref)
        x_3_out, y_3_out = rv_3.get_grid()
        assert_array_equal(x_3_out, x_ref)
        assert_array_equal(y_3_out, y_ref)
