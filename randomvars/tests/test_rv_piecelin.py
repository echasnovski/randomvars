# pylint: disable=missing-function-docstring
"""Tests for 'rv_piecelin.py' file
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from randomvars.rv_piecelin import rv_piecelin


def assert_equal_seq(first, second, *args, **kwargs):
    assert len(first) == len(second)
    for el1, el2 in zip(first, second):
        assert_array_equal(el1, el2, *args, **kwargs)


def assert_equal_rv_pieceilin(rv_p_1, rv_p_2):
    grid_1 = rv_p_1.x, rv_p_1.y, rv_p_1.p
    grid_2 = rv_p_2.x, rv_p_2.y, rv_p_2.p
    assert_equal_seq(grid_1, grid_2)


class TestRVPiecelin:
    """Regression tests for `rv_piecelin` class
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
            assert_equal_rv_pieceilin(rv, rv_ref)

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
        assert_equal_rv_pieceilin(rv_1, rv_ref)

        # Check if `y` is normalized
        rv_2 = rv_piecelin(x=x_ref, y=10 * y_ref)
        assert_equal_rv_pieceilin(rv_2, rv_ref)

        # Check if `x` and `y` are rearranged if not sorted
        rv_3 = rv_piecelin(x=x_ref[[1, 0, 2]], y=10 * y_ref[[1, 0, 2]])
        assert_equal_rv_pieceilin(rv_3, rv_ref)

    def test_xyp(self):
        """Tests for `x`, `y`, and `p` properties"""
        x = np.arange(11)
        y = np.repeat(0.1, 11)
        rv = rv_piecelin(x, y)

        assert_array_equal(rv.x, x)
        assert_array_equal(rv.y, y)
        assert_array_almost_equal(rv.p, np.arange(11) / 10, decimal=15)

    def test_pdf_coeffs(self):
        rv = rv_piecelin([0, 1, 2], [0, 1, 0])
        x = np.array([-1, 0, 0.5, 1, 1.5, 2, 2.5])

        with pytest.raises(ValueError, match="one of"):
            rv.pdf_coeffs(x, side="a")

        assert_equal_seq(
            rv.pdf_coeffs(x),
            (np.array([0, 0, 0, 2, 2, 2, 0]), np.array([0, 1, 1, -1, -1, -1, 0])),
        )
        assert_equal_seq(
            rv.pdf_coeffs(x, side="left"),
            (np.array([0, 0, 0, 0, 2, 2, 0]), np.array([0, 1, 1, 1, -1, -1, 0])),
        )
        assert_equal_seq(
            rv.pdf_coeffs(np.array([-np.inf, np.nan, np.inf])),
            (np.array([0, np.nan, 0]), np.array([0, np.nan, 0])),
        )

    def test_pdf(self):
        """Tests for `.pdf()` method, which logic is implemented in `._pdf()`
        """
        rv = rv_piecelin([0, 1, 3], [0.5, 0.5, 0])

        # Regular checks
        x = np.array([-1, 0, 0.5, 1, 2, 3, 4])
        assert_array_equal(rv.pdf(x), np.array([0, 0.5, 0.5, 0.5, 0.25, 0, 0]))

        # Input around edges
        x = np.array([0 - 1e-10, 0 + 1e-10, 3 - 1e-10, 3 + 1e-10])
        assert_array_almost_equal(
            rv.pdf(x), np.array([0, 0.5, 0.25e-10, 0]), decimal=12
        )

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.pdf(x), np.array([0, np.nan, 0]))

        # Dirac-like random variable
        rv_dirac = rv_piecelin([10 - 1e-8, 10, 10 + 1e-8], [0, 1, 0])
        x = np.array([10 - 1e-8, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + 1e-8])
        ## Accuracy is of order of 10 due to extreme magnitudes of values
        assert_array_almost_equal(
            rv_dirac.pdf(x), np.array([0, 0.5e8, 1e8, 0.5e8, 0]), decimal=-1
        )

    def test_cdf(self):
        """Tests for `.cdf()` method, which logic is implemented in `._cdf()`
        """
        rv_1 = rv_piecelin([0, 1, 2], [0, 1, 0])

        # Regular checks
        x = np.array([-1, 0, 0.5, 1, 1.5, 2, 3])
        assert_array_equal(rv_1.cdf(x), np.array([0, 0, 0.125, 0.5, 0.875, 1, 1]))

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv_1.cdf(x), np.array([0, np.nan, 1]))

        # Dirac-like random variable
        rv_dirac = rv_piecelin([10 - 1e-8, 10, 10 + 1e-8], [0, 1, 0])
        x = np.array([10 - 1e-8, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + 1e-8])
        assert_array_almost_equal(
            rv_dirac.cdf(x), np.array([0, 0.125, 0.5, 0.875, 1]), decimal=7
        )

    def test_ppf(self):
        """Tests for `.ppf()` method, which logic is implemented in `._cdf()`
        """
        # `ppf()` method should be inverse to `cdf()` for every sensible input
        rv_1 = rv_piecelin([0, 1, 2], [0, 1, 0])

        # Regular checks
        q = np.array([0, 0.125, 0.5, 0.875, 1])
        assert_array_equal(rv_1.ppf(q), np.array([0, 0.5, 1, 1.5, 2]))

        # Bad input
        q = np.array([-np.inf, -1e-8, np.nan, 1 + 1e-8, np.inf])
        assert_array_equal(
            rv_1.ppf(q), np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )

        # Dirac-like random variable
        rv_dirac = rv_piecelin([10 - 1e-8, 10, 10 + 1e-8], [0, 1, 0])
        q = np.array([0, 0.125, 0.5, 0.875, 1])
        assert_array_almost_equal(
            rv_dirac.ppf(q),
            np.array([10 - 1e-8, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + 1e-8]),
            decimal=9,
        )
