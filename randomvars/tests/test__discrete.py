# pylint: disable=missing-function-docstring
"""Tests for '_discrete.py' file"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from randomvars._discrete import Disc
import randomvars.options as op


def assert_equal_seq(first, second, *args, **kwargs):
    assert len(first) == len(second)
    for el1, el2 in zip(first, second):
        assert_array_equal(el1, el2, *args, **kwargs)


def assert_equal_disc(rv_1, rv_2):
    grid_1 = rv_1.x, rv_1.prob, rv_1.p
    grid_2 = rv_2.x, rv_2.prob, rv_2.p
    assert_equal_seq(grid_1, grid_2)


class TestDisc:
    """Regression tests for `Disc` class"""

    def test_init_errors(self):
        def check_one_input(def_args, var):
            with pytest.raises(ValueError, match=f"`{var}`.*numpy array"):
                def_args[var] = {"a": None}
                Disc(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*numeric"):
                def_args[var] = ["a", "a"]
                Disc(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.nan]
                Disc(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.inf]
                Disc(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*1d array"):
                def_args[var] = [[0, 1]]
                Disc(**def_args)

        check_one_input({"prob": [0.2, 0.8]}, "x")
        check_one_input({"x": [0, 1]}, "prob")

        with pytest.raises(ValueError, match="[Ll]engths.*match"):
            Disc([0, 1], [1, 1, 1])

        with pytest.warns(UserWarning, match="`x`.*not sorted.*`x` and `prob`"):
            rv = Disc([1, 0], [0.2, 0.8])
            rv_ref = Disc([0, 1], [0.8, 0.2])
            assert_equal_disc(rv, rv_ref)

        with pytest.raises(ValueError, match="`prob`.*negative"):
            Disc([0, 1], [0.8, -1])

        with pytest.raises(ValueError, match="`prob`.*no positive"):
            Disc([0, 1], [0, 0])

    def test_init(self):
        x_ref = np.array([0, 1, 2])
        prob_ref = np.array([0.1, 0.2, 0.7])
        rv_ref = Disc(x_ref, prob_ref)

        # Simple case with non-numpy input
        rv_1 = Disc(x=x_ref.tolist(), prob=prob_ref.tolist())
        assert_equal_disc(rv_1, rv_ref)

        # Check if `prob` is normalized
        rv_2 = Disc(x=x_ref, prob=10 * prob_ref)
        assert_equal_disc(rv_2, rv_ref)

        # Check if `x` and `prob` are rearranged if not sorted
        with pytest.warns(UserWarning, match="`x`.*not sorted"):
            rv_3 = Disc(x=x_ref[[1, 0, 2]], prob=prob_ref[[1, 0, 2]])
        assert_equal_disc(rv_3, rv_ref)

    def test_xprobp(self):
        """Tests for `x`, `prob`, and `p` properties"""
        x = np.arange(10)
        prob = np.repeat(0.1, 10)
        rv = Disc(x, prob)

        assert_array_equal(rv.x, x)
        assert_array_equal(rv.prob, prob)
        assert_array_equal(rv.p, np.cumsum(prob))

    def test_pmf(self):
        """Tests for `.pmf()` method, which logic is implemented in `._pmf()`"""
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        rtol, atol = op.get_option("tolerance")

        # Regular checks
        x = np.array([0, 0.5, 1, 3, (1 + rtol) * 3 + 0.5 * atol])
        assert_array_equal(rv.pmf(x), np.array([0, 0.1, 0.2, 0.7, 0.7]))

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.pmf(x), np.array([0, np.nan, 0]))

        # Using tolerance option
        with op.option_context({"tolerance": (0, 1e-10)}):
            assert_array_equal(rv.pmf([1 + 1.01e-10, 1 + 0.9e-10]), [0.0, 0.2])

        with op.option_context({"tolerance": (1e-2, 1e-10)}):
            assert_array_equal(
                rv.pmf([(1 + 1e-2) * 1 + 1.01e-10, (1 + 1e-2) * 1 + 0.99e-10]),
                [0.0, 0.2],
            )

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_equal(rv.pmf(x), np.array([[0.0, 0.1], [0.0, 0.0]]))

    def test_cdf(self):
        """Tests for `.cdf()` method, which logic is implemented in `._cdf()`"""
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        h = 1e-12

        # Regular checks
        x = np.array([-10, 0.5 - h, 0.5, 0.5 + h, 1 - h, 1, 1 + h, 3 - h, 3, 3 + h, 10])
        assert_array_almost_equal(
            rv.cdf(x),
            np.array([0, 0, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1, 1, 1]),
            decimal=12,
        )

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.cdf(x), np.array([0, np.nan, 1]))

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_almost_equal(
            rv.cdf(x), np.array([[0.0, 0.1], [0.3, 1.0]]), decimal=12
        )

    def test_ppf(self):
        """Tests for `.ppf()` method, which logic is implemented in `._ppf()`"""
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        h = 1e-12

        # Regular checks
        ## Outputs for q=0 and q=1 should be equal to minimum and maximum elements
        q = np.array([0, 0.1 - h, 0.1, 0.1 + h, 0.3 - h, 0.3, 0.3 + h, 1 - h, 1])
        assert_array_equal(rv.ppf(q), np.array([0.5, 0.5, 0.5, 1, 1, 1, 3, 3, 3]))

        # Bad input
        q = np.array([-np.inf, -h, np.nan, 1 + h, np.inf])
        assert_array_equal(
            rv.ppf(q), np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )

        # Broadcasting
        q = np.array([[0, 0.5], [0.0, 1.0]])
        assert_array_equal(rv.ppf(q), np.array([[0.5, 3], [0.5, 3]]))

    def test_rvs(self):
        """Tests for `.rvs()`"""
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])

        # Regular checks
        smpl = rv.rvs(size=100)
        assert np.all((rv.a <= smpl) & (smpl <= rv.b))

        # Treats default `size` as 1
        assert rv.rvs().shape == tuple()

        # Broadcasting
        smpl_array = rv.rvs(size=(10, 2))
        assert smpl_array.shape == (10, 2)

        # Usage of `random_state`
        smpl_1 = rv.rvs(size=100, random_state=np.random.RandomState(101))
        smpl_2 = rv.rvs(size=100, random_state=np.random.RandomState(101))
        assert_array_equal(smpl_1, smpl_2)

        # Usage of integer `random_state` as a seed
        smpl_1 = rv.rvs(size=100, random_state=101)
        smpl_2 = rv.rvs(size=100, random_state=101)
        assert_array_equal(smpl_1, smpl_2)
