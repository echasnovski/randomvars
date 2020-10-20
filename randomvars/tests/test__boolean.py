# pylint: disable=missing-function-docstring
"""Tests for '_boolean.py' file"""
import numpy as np
from numpy.testing import assert_array_equal
import scipy.stats.distributions as distrs
import pytest

from randomvars._boolean import Bool
from randomvars._discrete import Disc
from randomvars._utils import _assert_equal_seq
import randomvars.options as op


def assert_equal_bool(rv_1, rv_2):
    grid_1 = rv_1.prob_false, rv_1.prob_true
    grid_2 = rv_2.prob_false, rv_2.prob_true
    _assert_equal_seq(grid_1, grid_2)


class TestBool:
    """Regression tests for `Bool` class"""

    def test_init_errors(self):
        with pytest.raises(ValueError, match="number"):
            Bool("a")
        with pytest.raises(ValueError, match="0"):
            Bool(-0.1)
        with pytest.raises(ValueError, match="1"):
            Bool(1.1)

    def test_init(self):
        # Basic usage
        rv_out = Bool(prob_true=0.75)
        assert rv_out.prob_true == 0.75
        assert rv_out.prob_false == 0.25

        # Integer edge cases
        rv_out = Bool(prob_true=0)
        assert rv_out.prob_true == 0.0
        assert rv_out.prob_false == 1.0

        rv_out = Bool(prob_true=1)
        assert rv_out.prob_true == 1.0
        assert rv_out.prob_false == 0.0

    def test_str(self):
        rv = Bool(0.75)
        assert str(rv) == "Boolean RV with 0.75 probability of True"

    def test_properties(self):
        """Tests for properties"""
        prob_true = 0.75
        rv = Bool(prob_true)

        assert_array_equal(rv.prob_false, 1 - prob_true)
        assert_array_equal(rv.prob_true, prob_true)

    def test_from_rv_basic(self):
        prob_true = 0.75
        rv = distrs.rv_discrete(values=([0, 1], [1 - prob_true, prob_true]))
        rv_out = Bool.from_rv(rv)
        rv_ref = Bool(prob_true=prob_true)
        assert_equal_bool(rv_out, rv_ref)

        # Object of `Bool` class should be returned untouched
        rv = Bool(prob_true)
        rv.aaa = "Extra method"
        rv2 = Bool.from_rv(rv)
        assert_equal_bool(rv, rv2)
        assert "aaa" in dir(rv2)

        # Works with rv with not only 0 and 1 values
        rv_many = distrs.rv_discrete(values=([-1, 0, 1], [0.5, 0.375, 0.125]))
        rv_out = Bool.from_rv(rv_many)
        ## Only probability of 0 should matter
        rv_ref = Bool(prob_true=1 - 0.375)
        assert_equal_bool(rv_out, rv_ref)

    def test_from_rv_errors(self):
        # Absence of `cdf` method should result intro error
        tmp = object()
        with pytest.raises(ValueError, match="cdf"):
            Bool.from_rv(tmp)

    def test_from_rv_options(self):
        # Usage of `tolerance` option
        x = [-1e-2, 0, 1e-3]
        prob = [0.125, 0.5, 0.375]
        rv = distrs.rv_discrete(values=(x, prob))

        with op.option_context({"tolerance": (0, 1e-4)}):
            assert_equal_bool(Bool.from_rv(rv), Bool(prob_true=1 - 0.5))
        with op.option_context({"tolerance": (0, 5e-3)}):
            # Close positive values shouldn't affect output, because it is
            # computed using `cdf(0) - cdf(-atol)`
            assert_equal_bool(Bool.from_rv(rv), Bool(prob_true=1 - 0.5))
        with op.option_context({"tolerance": (0, 5e-2)}):
            assert_equal_bool(Bool.from_rv(rv), Bool(prob_true=1 - 0.625))

    def test_from_sample_basic(self):
        # Normal usage
        x = np.array([True, False, False, True, True])

        rv = Bool.from_sample(x)
        rv_ref = Bool(prob_true=0.6)
        assert isinstance(rv, Bool)
        assert_equal_bool(rv, rv_ref)

        # Accepting other types
        x = np.array([0, 1, 2, 1])
        rv = Bool.from_sample(x)
        rv_ref = Bool.from_sample(x.astype("bool"))
        assert_equal_bool(rv, rv_ref)

    def test_from_sample_errors(self):
        # As everything is convertible to boolean array, no error can be thrown
        # regarding input type

        with pytest.raises(ValueError, match="1d"):
            Bool.from_sample([[True], [False]])

    def test_from_sample_options(self):
        x = [True, False, False, True]

        # "boolean_estimator"
        with op.option_context({"boolean_estimator": lambda x: 0}):
            rv = Bool.from_sample(x)
            assert_equal_bool(rv, Bool(prob_true=0))

        # "boolean_estimator" which returns allowed classes
        ## `Bool` object should be returned untouched
        rv_estimation = Bool(prob_true=1)
        rv_estimation.aaa = "Extra method"
        with op.option_context({"boolean_estimator": lambda x: rv_estimation}):
            rv = Bool.from_sample(x)
            assert "aaa" in dir(rv)

        ## `Disc` and "scipy" distribution should be forwarded to
        ## `Bool.from_rv()`
        rv_disc = Disc(x=[0, 1], prob=[0.125, 0.875])
        with op.option_context({"boolean_estimator": lambda x: rv_disc}):
            rv = Bool.from_sample(x)
            rv_ref = Bool.from_rv(rv_disc)
            assert_equal_bool(rv, rv_ref)

        rv_bernoulli = distrs.bernoulli(p=0.625)
        with op.option_context({"boolean_estimator": lambda x: rv_bernoulli}):
            rv = Bool.from_sample(x)
            rv_ref = Bool.from_rv(rv_bernoulli)
            assert_equal_bool(rv, rv_ref)

    def test_pmf(self):
        """Tests for `.pmf()` method"""
        rv = Bool(0.75)

        # Normal usage
        x = [False, False, True, True]
        assert_array_equal(rv.pmf(x), [0.25, 0.25, 0.75, 0.75])

        # Other types
        x = np.asarray([-1, -1e-12, 0, 0.5, 1, 2])
        assert_array_equal(rv.pmf(x), rv.pmf(x.astype("bool")))

        x = np.asarray([lambda x: x, {"a": 1}, {}])
        assert_array_equal(rv.pmf(x), rv.pmf(x.astype("bool")))

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.pmf(x), rv.pmf(x.astype("bool")))

        # Broadcasting
        x = np.array([[False, True], [True, False]])
        assert_array_equal(rv.pmf(x), np.array([[0.25, 0.75], [0.75, 0.25]]))

    def test_cdf(self):
        """Tests for `.cdf()` method"""
        rv = Bool(0.75)

        # Normal usage
        x = [False, False, True, True]
        assert_array_equal(rv.cdf(x), [0.25, 0.25, 1.0, 1.0])

        # Other types
        x = np.array([-1, -1e-12, 0, 0.5, 1, 2])
        assert_array_equal(rv.cdf(x), rv.cdf(x.astype("bool")))

        x = np.asarray([lambda x: x, {"a": 1}, {}])
        assert_array_equal(rv.cdf(x), rv.cdf(x.astype("bool")))

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.cdf(x), rv.cdf(x.astype("bool")))

        # Broadcasting
        x = np.array([[False, True], [True, False]])
        assert_array_equal(rv.cdf(x), np.array([[0.25, 1.0], [1.0, 0.25]]))

    def test_ppf(self):
        """Tests for `.ppf()` method"""
        rv = Bool(0.75)
        h = 1e-12

        # Normal usage
        q = np.array([0, 0.25 - h, 0.25, 0.25 + h, 1 - h, 1])
        out = rv.ppf(q)
        assert_array_equal(out, np.array([False, False, False, True, True, True]))
        assert out.dtype == np.dtype("bool")

        # Bad input will result into `True` instead of `numpy.nan` as this is
        # how Numpy converts `numpy.nan` to "bool" dtype
        q = np.array([-np.inf, -h, np.nan, 1 + h, np.inf])
        out = rv.ppf(q)
        assert_array_equal(out, np.array([True, True, True, True, True]))
        assert out.dtype == np.dtype("bool")

        # Broadcasting
        q = np.array([[0.0, 0.5], [0.0, 1.0]])
        out = rv.ppf(q)
        assert_array_equal(out, np.array([[False, True], [False, True]]))
        assert out.dtype == np.dtype("bool")

    def test_rvs(self):
        """Tests for `.rvs()`"""
        rv = Bool(0.75)

        # Regular checks
        ## Output should be boolean
        smpl = rv.rvs(size=100, random_state=101)
        assert_array_equal(np.unique(smpl), [False, True])
        assert smpl.dtype == np.dtype("bool")

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
