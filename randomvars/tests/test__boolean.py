# pylint: disable=missing-function-docstring
"""Tests for '_boolean.py' file"""
import numpy as np
from numpy.testing import assert_array_equal
import scipy.stats.distributions as distrs
import pytest

from randomvars._boolean import Bool
from randomvars.tests.commontests import (
    h,
    _test_equal_rand,
    _test_input_coercion,
    _test_from_rv_rand,
    _test_from_sample_rand,
    _test_log_fun,
    _test_one_value_input,
    _test_rvs_method,
)
import randomvars.options as op


class TestBool:
    """Regression tests for `Bool` class"""

    def test_init_errors(self):
        with pytest.raises(TypeError, match="number"):
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
        prob_true = 0.75
        rv = Bool(prob_true)

        assert rv.params == {"prob_true": rv.prob_true}

        assert_array_equal(rv.prob_false, 1 - prob_true)
        assert_array_equal(rv.prob_true, prob_true)
        assert_array_equal(rv.a, False)
        assert_array_equal(rv.b, True)

    def test_support(self):
        rv = Bool(0.75)
        assert rv.support() == (False, True)

    def test_compress(self):
        rv = Bool(0.75)
        assert rv.compress() is rv

    def test_from_rv_basic(self):
        prob_true = 0.75
        rv = distrs.rv_discrete(values=([0, 1], [1 - prob_true, prob_true]))
        rv_out = Bool.from_rv(rv)
        rv_ref = Bool(prob_true=prob_true)
        _test_equal_rand(rv_out, rv_ref)

        # Objects of `Rand` class should be `convert()`ed
        _test_from_rv_rand(cls=Bool, to_class="Bool")

        # Works with rv with not only 0 and 1 values
        rv_many = distrs.rv_discrete(values=([-1, 0, 1], [0.5, 0.375, 0.125]))
        rv_out = Bool.from_rv(rv_many)
        ## Only probability of 0 should matter
        rv_ref = Bool(prob_true=1 - 0.375)
        _test_equal_rand(rv_out, rv_ref)

    def test_from_rv_errors(self):
        # Absence of `cdf` method should result intro error
        tmp = object()
        with pytest.raises(ValueError, match="cdf"):
            Bool.from_rv(tmp)

    def test_from_rv_options(self):
        # Usage of `base_tolerance` option
        x = [-1e-2, 0, 1e-3]
        prob = [0.125, 0.5, 0.375]
        rv = distrs.rv_discrete(values=(x, prob))

        with op.option_context({"base_tolerance": 1e-4}):
            _test_equal_rand(Bool.from_rv(rv), Bool(prob_true=1 - 0.5))
        with op.option_context({"base_tolerance": 5e-3}):
            # Close positive values shouldn't affect output, because it is
            # computed using `cdf(0) - cdf(-base_tol)`
            _test_equal_rand(Bool.from_rv(rv), Bool(prob_true=1 - 0.5))
        with op.option_context({"base_tolerance": 5e-2}):
            _test_equal_rand(Bool.from_rv(rv), Bool(prob_true=1 - 0.625))

    def test_from_sample_basic(self):
        # Normal usage
        x = np.array([True, False, False, True, True])

        rv = Bool.from_sample(x)
        rv_ref = Bool(prob_true=0.6)
        assert isinstance(rv, Bool)
        _test_equal_rand(rv, rv_ref)

        # Accepting other types
        x = np.array([0, 1, 2, 1])
        rv = Bool.from_sample(x)
        rv_ref = Bool.from_sample(x.astype("bool"))
        _test_equal_rand(rv, rv_ref)

    def test_from_sample_errors(self):
        # As everything is convertible to boolean array, no error can be thrown
        # regarding input type

        with pytest.raises(ValueError, match="1d"):
            Bool.from_sample([[True], [False]])

    def test_from_sample_options(self):
        x = [True, False, False, True]

        # "estimator_bool"
        with op.option_context({"estimator_bool": lambda x: 0}):
            rv = Bool.from_sample(x)
            _test_equal_rand(rv, Bool(prob_true=0))

        # "estimator_bool" which returns allowed classes
        ## `Rand` class should be forwarded to `from_rv()` method
        _test_from_sample_rand(
            cls=Bool,
            sample=x,
            estimator_option="estimator_bool",
        )

        ## "Scipy" distributions should be forwarded to `Bool.from_rv()`
        rv_bernoulli = distrs.bernoulli(p=0.625)
        with op.option_context({"estimator_bool": lambda x: rv_bernoulli}):
            _test_equal_rand(Bool.from_sample(x), Bool.from_rv(rv_bernoulli))

    def test_pdf(self):
        rv = Bool(0.75)
        with pytest.raises(AttributeError, match=r"Use `pmf\(\)`"):
            rv.pdf(False)

    def test_logpdf(self):
        rv = Bool(0.75)
        with pytest.raises(AttributeError, match=r"Use `logpmf\(\)`"):
            rv.logpdf(False)

    def test_pmf(self):
        rv = Bool(0.75)

        # Normal usage
        x = [False, False, True, True]
        assert_array_equal(rv.pmf(x), [0.25, 0.25, 0.75, 0.75])

        # Coercion of not ndarray input
        _test_input_coercion(rv.pmf, x)

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

        # One value input
        _test_one_value_input(rv.pmf, True)
        _test_one_value_input(rv.pmf, np.nan)

    def test_logpmf(self):
        rv = Bool(0.75)
        x_ref = [-1, 0, 1, 3, np.inf, np.nan]
        _test_log_fun(rv.logpmf, rv.pmf, x_ref=[-1, 0, 1, 3, np.inf, np.nan])

    def test_cdf(self):
        rv = Bool(0.75)

        # Normal usage
        x = [False, False, True, True]
        assert_array_equal(rv.cdf(x), [0.25, 0.25, 1.0, 1.0])

        # Coercion of not ndarray input
        _test_input_coercion(rv.cdf, x)

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

        # One value input
        _test_one_value_input(rv.cdf, True)
        _test_one_value_input(rv.cdf, np.nan)

    def test_logcdf(self):
        rv = Bool(0.75)
        _test_log_fun(rv.logcdf, rv.cdf, x_ref=[-1, 0, 1, 3, np.inf, np.nan])

    def test_sf(self):
        rv = Bool(0.75)
        x_ref = [-1, 0, 1, 3, np.inf, np.nan]
        assert_array_equal(rv.sf(x_ref), 1 - rv.cdf(x_ref))

    def test_logsf(self):
        rv = Bool(0.75)
        _test_log_fun(rv.logsf, rv.sf, x_ref=[-1, 0, 1, 3, np.inf, np.nan])

    def test_ppf(self):
        rv = Bool(0.75)

        # Normal usage
        q = np.array([0, 0.25 - h, 0.25, 0.25 + h, 1 - h, 1])
        out = rv.ppf(q)
        assert_array_equal(out, np.array([False, False, False, True, True, True]))
        assert out.dtype == np.dtype("bool")

        # Coercion of not ndarray input
        _test_input_coercion(rv.ppf, q)

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

        # One value input
        _test_one_value_input(rv.ppf, 0.25)
        _test_one_value_input(rv.ppf, -1)
        _test_one_value_input(rv.ppf, np.nan)

    def test_isf(self):
        rv = Bool(0.75)

        q = np.array([0, 0.75 - h, 0.75, 0.75 + h, 1 - h, 1])
        out = rv.isf(q)
        # Output is lowest value `t` (within supp.) for which `P(X > t) <= q`
        assert_array_equal(out, rv.ppf(1 - q))
        assert out.dtype == np.dtype("bool")

    def test_rvs(self):
        rv = Bool(0.75)

        _test_rvs_method(rv)

        ## Output should be boolean
        smpl = rv.rvs(size=100, random_state=101)
        assert_array_equal(np.unique(smpl), [False, True])
        assert smpl.dtype == np.dtype("bool")

    def test_integrate_cdf(self):
        rv = Bool(prob_true=0.75)
        assert np.allclose(rv.integrate_cdf(-10, 10), 0.25 * 1 + 1 * 9)

    def test_convert(self):
        import randomvars._continuous as cont
        import randomvars._discrete as disc
        import randomvars._mixture as mixt

        rv = Bool(prob_true=0.75)

        # By default and supplying `None` should return self
        assert rv.convert() is rv
        assert rv.convert(None) is rv

        # Converting to own class should return self
        out_bool = rv.convert("Bool")
        assert out_bool is rv

        # Converting to Cont should result into conversion to Disc and then to Cont
        out_cont = rv.convert("Cont")
        out_ref = rv.convert("Disc").convert("Cont")
        assert isinstance(out_cont, cont.Cont)
        assert_array_equal(out_cont.x, out_ref.x)
        assert_array_equal(out_cont.y, out_ref.y)

        # Converting to Disc should result into discrete RV with `x=[0, 1],
        # p=[prob_false, prob_true]`
        out_disc = rv.convert("Disc")
        assert isinstance(out_disc, disc.Disc)
        assert_array_equal(out_disc.x, [0, 1])
        assert_array_equal(out_disc.p, [rv.prob_false, rv.prob_true])

        # Converting to Mixt should result into degenerate mixture with only
        # discrete component (which is a conversion of input to Disc)
        out_mixt = rv.convert("Mixt")
        out_disc_ref = rv.convert("Disc")
        assert isinstance(out_mixt, mixt.Mixt)
        assert_array_equal(out_mixt.disc.x, out_disc_ref.x)
        assert_array_equal(out_mixt.disc.p, out_disc_ref.p)
        assert out_mixt.weight_disc == 1.0

        # Any other target class should result into error
        with pytest.raises(ValueError, match="one of"):
            rv.convert("aaa")
