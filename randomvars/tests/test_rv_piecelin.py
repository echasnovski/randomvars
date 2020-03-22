# pylint: disable=missing-function-docstring
"""Tests for 'rv_piecelin.py' file
"""
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from randomvars.rv_piecelin import rv_piecelin


def assert_equal_grid(rv1, rv2, *args, **kwargs):
    grid1 = rv1.get_grid()
    grid2 = rv2.get_grid()
    assert_array_equal(grid1[0], grid2[0], *args, **kwargs)
    assert_array_equal(grid1[1], grid2[1], *args, **kwargs)


class TestRVPiecelin:
    """Tests for `rv_piecelin` class
    """

    def test_init_errors(self):
        def check_one_input(def_args, var):
            with pytest.raises(ValueError, match=f"`{var}`.*numpy array"):
                def_args[var] = {"a": None}
                rv_piecelin(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*numeric"):
                def_args[var] = ["a", "a"]
                rv_piecelin(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.nan]
                rv_piecelin(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.inf]
                rv_piecelin(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*1d array"):
                def_args[var] = [[0, 1]]
                rv_piecelin(**def_args)

        check_one_input({"y": [1, 1]}, "x")
        check_one_input({"x": [0, 1]}, "y")

        with pytest.raises(ValueError, match="[Ll]engths.*match"):
            rv_piecelin([0, 1], [1, 1, 1])

        with pytest.warns(UserWarning, match="`x`.*not sorted"):
            rv = rv_piecelin([1, 0], [0, 2])
            rv_ref = rv_piecelin([0, 1], [2, 0])
            assert_equal_grid(rv, rv_ref)

        with pytest.raises(ValueError, match="`y`.*negative"):
            rv_piecelin([0, 1], [1, -1])

        with pytest.raises(ValueError, match="`y`.*no positive"):
            rv_piecelin([0, 1], [0, 0])

    def test_init(self):
        x_ref = np.array([0, 1, 2])
        y_ref = np.array([0, 1, 0])
        rv_ref = rv_piecelin(x_ref, y_ref)

        # Simple case with non-numpy input
        rv_1 = rv_piecelin(x=x_ref.tolist(), y=y_ref.tolist())
        assert_equal_grid(rv_1, rv_ref)

        # Check if `y` is normalized
        rv_2 = rv_piecelin(x=x_ref, y=10 * y_ref)
        assert_equal_grid(rv_2, rv_ref)

        # Check if `x` and `y` are rearranged if not sorted
        rv_3 = rv_piecelin(x=x_ref[[1, 0, 2]], y=10 * y_ref[[1, 0, 2]])
        assert_equal_grid(rv_3, rv_ref)
