# pylint: disable=missing-function-docstring
"""Tests for '_discrete.py' file"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.stats.distributions as distrs
import pytest

from randomvars._discrete import Disc
from randomvars._utils import _tolerance
from randomvars.tests.commontests import (
    DECIMAL,
    h,
    _test_equal_rand,
    _test_input_coercion,
    _test_from_rv_rand,
    _test_from_sample_rand,
    _test_log_fun,
    _test_one_value_input,
    _test_rvs_method,
)
from randomvars.options import config

DISTRIBUTIONS_FINITE = {
    "fin_bernoulli": distrs.bernoulli(p=0.9),
    "fin_binom": distrs.binom(n=10, p=0.7),
    "fin_binom_wide": distrs.binom(n=101, p=0.7),
    "fin_randint": distrs.randint(low=-5, high=4),
}

DISTRIBUTIONS_INFINITE = {
    "inf_geom": distrs.geom(p=0.1),
    "inf_poisson": distrs.poisson(mu=5),
    "inf_poisson_big": distrs.poisson(mu=1001),
}

DISTRIBUTIONS_SHIFTED = {
    "shift_binom": distrs.binom(n=10, p=0.7, loc=-np.pi),
    "shift_randint": distrs.randint(low=-5, high=4, loc=np.pi),
}

DISTRIBUTIONS = {
    **DISTRIBUTIONS_FINITE,
    **DISTRIBUTIONS_INFINITE,
    **DISTRIBUTIONS_SHIFTED,
}


class TestDisc:
    """Regression tests for `Disc` class"""

    def test_init_errors(self):
        def check_one_input(def_args, var):
            with pytest.raises(TypeError, match=f"`{var}`.*numpy array"):
                def_args[var] = {"a": None}
                Disc(**def_args)
            with pytest.raises(TypeError, match=f"`{var}`.*float"):
                def_args[var] = ["a", "a"]
                Disc(**def_args)
            with pytest.raises(TypeError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.nan]
                Disc(**def_args)
            with pytest.raises(TypeError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.inf]
                Disc(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*1d array"):
                def_args[var] = [[0, 1]]
                Disc(**def_args)

        check_one_input({"p": [0.2, 0.8]}, "x")
        check_one_input({"x": [0, 1]}, "p")

        with pytest.raises(ValueError, match="[Ll]engths.*match"):
            Disc([0, 1], [1, 1, 1])

        with pytest.warns(UserWarning, match="`x`.*not sorted.*`x` and `p`"):
            rv = Disc([1, 0], [0.2, 0.8])
            rv_ref = Disc([0, 1], [0.8, 0.2])
            _test_equal_rand(rv, rv_ref)

        with pytest.raises(ValueError, match="`p`.*negative"):
            Disc([0, 1], [0.8, -1])

        with pytest.raises(ValueError, match="`p`.*no positive"):
            Disc([0, 1], [0, 0])

    def test_init(self):
        x_ref = np.array([0.1, 1, 2])
        p_ref = np.array([0.1, 0.2, 0.7])
        rv_ref = Disc(x_ref, p_ref)

        # Simple case with non-numpy input
        rv_1 = Disc(x=x_ref.tolist(), p=p_ref.tolist())
        _test_equal_rand(rv_1, rv_ref)

        # Check if `p` is normalized
        rv_2 = Disc(x=x_ref, p=10 * p_ref)
        _test_equal_rand(rv_2, rv_ref)

        # Check that zero probability is allowed
        rv_3 = Disc(x=[0, 1, 3], p=[0, 0.5, 0.5])
        assert_array_equal(rv_3.x, np.array([0, 1, 3]))
        assert_array_equal(rv_3.p, np.array([0, 0.5, 0.5]))

        # Check if `x` and `p` are rearranged if not sorted
        with pytest.warns(UserWarning, match="`x`.*not sorted"):
            rv_4 = Disc(x=x_ref[[1, 0, 2]], p=p_ref[[1, 0, 2]])
            _test_equal_rand(rv_4, rv_ref)

        # Check if duplicated values are removed from `x`
        with pytest.warns(UserWarning, match="duplicated"):
            # First pair of xy-grid is taken among duplicates
            rv_5 = Disc(x=x_ref[[0, 1, 1, 2]], p=p_ref[[0, 1, 2, 2]])
            _test_equal_rand(rv_5, rv_ref)

    def test_str(self):
        rv = Disc([0, 2, 4], [0.125, 0, 0.875])
        assert str(rv) == "Discrete RV with 3 values (support: [0.0, 4.0])"

        # Uses singular noun with one value
        rv = Disc([1], [1])
        assert str(rv) == "Discrete RV with 1 value (support: [1.0, 1.0])"

    def test_properties(self):
        x = np.arange(10)
        p = np.repeat(0.1, 10)
        rv = Disc(x, p)

        assert list(rv.params.keys()) == ["x", "p"]
        assert_array_equal(rv.params["x"], x)
        assert_array_equal(rv.params["p"], p)

        assert_array_equal(rv.x, x)
        assert_array_equal(rv.p, p)
        assert rv.a == 0
        assert rv.b == 9

    def test_support(self):
        rv = Disc([0.5, 1.5, 4.5], [0.25, 0.375, 0.375])
        assert rv.support() == (0.5, 4.5)

    def test_compress(self):
        # Keep only x-values with positive probabilities
        _test_equal_rand(
            Disc([0, 1, 2, 3], [0.0, 0.5, 0.0, 0.5]).compress(),
            Disc([1, 3], [0.5, 0.5]),
        )

        # If nothing to compress, self should be returned
        rv = Disc([0, 1], [0.5, 0.5])
        assert rv.compress() is rv

    def test_from_rv_basic(self):
        x = [0, 1, 5]
        p = [0.1, 0.4, 0.5]
        rv = distrs.rv_discrete(values=(x, p))
        rv_out = Disc.from_rv(rv)
        rv_ref = Disc(x=x, p=p)
        _test_equal_rand(rv_out, rv_ref)

        # Objects of `Rand` class should be `convert()`ed
        _test_from_rv_rand(cls=Disc, to_class="Disc")

        # Works with single-valued rv
        rv_single = distrs.rv_discrete(values=(2, 1))
        rv_out = Disc.from_rv(rv_single)
        assert_array_equal(rv_out.x, [2])

    def test_from_rv_errors(self):
        # Absence of either `cdf` or `ppf` method should result intro error
        class Tmp:
            pass

        tmp1 = Tmp()
        tmp1.ppf = lambda x: np.where((0 <= x) & (x <= 1), 1, 0)
        with pytest.raises(ValueError, match="cdf"):
            Disc.from_rv(tmp1)

        tmp2 = Tmp()
        tmp2.cdf = lambda x: np.where((0 <= x) & (x <= 1), 1, 0)
        with pytest.raises(ValueError, match="ppf"):
            Disc.from_rv(tmp2)

        # Having bad `cdf()` or `ppf()` that result into infinite loop should
        # result into error
        tmp3 = Tmp()
        tmp3.cdf = lambda x: x
        tmp3.ppf = lambda x: 0.5
        with pytest.raises(ValueError, match="Couldn't get increase"):
            Disc.from_rv(tmp3)

    def test_from_rv_options(self):
        # Usage of `small_prob` option
        x = [1, 2, 3]
        p = [0.5, 0.125, 0.375]
        rv = distrs.rv_discrete(values=(x, p))

        with config.context({"small_prob": 0.125 + 1e-5}):
            rv_out = Disc.from_rv(rv)
            rv_ref = Disc([1, 3], [0.5, 0.5])
            _test_equal_rand(rv_out, rv_ref)

    def test_from_sample_basic(self):
        x = np.array([0.1, -100, 1, np.pi, np.pi, 1, 3, 0.1])

        rv = Disc.from_sample(x)
        rv_ref = Disc(x=[-100, 0.1, 1, 3, np.pi], p=[0.125, 0.25, 0.25, 0.125, 0.25])
        assert isinstance(rv, Disc)
        _test_equal_rand(rv, rv_ref)

    def test_from_sample_errors(self):
        with pytest.raises(TypeError, match="numpy array with float"):
            Disc.from_sample(["a"])

        with pytest.raises(ValueError, match="1d"):
            Disc.from_sample([[1], [2]])

        # Warning is given with default discrete estimator if not all values
        # are finite
        with pytest.warns(UserWarning, match="has non-finite"):
            rv = Disc.from_sample([1, 2, np.nan])
            rv_ref = Disc(x=[1, 2], p=[0.5, 0.5])
            _test_equal_rand(rv, rv_ref)

        # Error is given with default discrete estimator if there is no finite
        # values
        with pytest.raises(ValueError, match="doesn't have finite values"):
            Disc.from_sample([-np.inf, np.nan, np.inf])

    def test_from_sample_options(self):
        binom = distrs.binom(n=10, p=0.5)

        rng = np.random.default_rng(101)
        x = binom.rvs(100, random_state=rng)

        # "estimator_disc"
        def single_value_estimator(x):
            return np.array([1.0]), np.array([1.0])

        with config.context({"estimator_disc": single_value_estimator}):
            rv = Disc.from_sample(x)
            _test_equal_rand(rv, Disc(x=[1.0], p=[1.0]))

        # "estimator_disc" which returns allowed classes
        ## `Rand` class should be forwarded to `from_rv()` method
        _test_from_sample_rand(
            cls=Disc,
            sample=x,
            estimator_option="estimator_disc",
        )

        ## "Scipy" distribution should be forwarded to `Disc.from_rv()`
        rv_binom = distrs.binom(n=10, p=0.5)
        with config.context({"estimator_disc": lambda x: rv_binom}):
            rv = Disc.from_sample(np.asarray([0, 1, 2]))
            rv_ref = Disc.from_rv(rv_binom)
            _test_equal_rand(rv, rv_ref)

    def test_pdf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        with pytest.raises(AttributeError, match=r"Use `pmf\(\)`"):
            rv.pdf(0.5)

    def test_logpdf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        with pytest.raises(AttributeError, match=r"Use `logpmf\(\)`"):
            rv.logpdf(0.5)

    def test_pmf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])

        # Regular checks
        x = np.array([0, 0.5, 1, 3, 3 + 0.5 * _tolerance(3), 3 + 1.5 * _tolerance(3)])
        assert_array_equal(rv.pmf(x), np.array([0.0, 0.1, 0.2, 0.7, 0.7, 0.0]))

        # Coercion of not ndarray input
        _test_input_coercion(rv.pmf, x)

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.pmf(x), np.array([0, np.nan, 0]))

        # Using different tolerance
        with config.context({"base_tolerance": 0.1}):
            assert_array_equal(rv.pmf([1 + 0.11, 1 + 0.09]), [0.0, 0.2])

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_equal(rv.pmf(x), np.array([[0.0, 0.1], [0.0, 0.0]]))

        # One value input
        _test_one_value_input(rv.pmf, 0.5)
        _test_one_value_input(rv.pmf, -1)
        _test_one_value_input(rv.pmf, np.nan)

    def test_logpmf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        _test_log_fun(rv.logpmf, rv.pmf, x_ref=[-1, 0.5, 3, np.inf, np.nan])

    def test_cdf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])

        # Regular checks
        x = np.array([-10, 0.5 - h, 0.5, 0.5 + h, 1 - h, 1, 1 + h, 3 - h, 3, 3 + h, 10])
        assert_array_almost_equal(
            rv.cdf(x),
            np.array([0, 0, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 1, 1, 1]),
            decimal=DECIMAL,
        )

        # Coercion of not ndarray input
        _test_input_coercion(rv.cdf, x)

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.cdf(x), np.array([0, np.nan, 1]))

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_almost_equal(
            rv.cdf(x), np.array([[0.0, 0.1], [0.3, 1.0]]), decimal=DECIMAL
        )

        # One value input
        _test_one_value_input(rv.cdf, 0.5)
        _test_one_value_input(rv.cdf, -1)
        _test_one_value_input(rv.cdf, np.nan)

    def test_logcdf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        _test_log_fun(rv.logcdf, rv.cdf, x_ref=[-1, 0.5, 3, np.inf, np.nan])

    def test_sf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        x_ref = [-1, 0.5, 3, np.inf, np.nan]
        assert_array_equal(rv.sf(x_ref), 1 - rv.cdf(x_ref))

    def test_logsf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        _test_log_fun(rv.logsf, rv.sf, x_ref=[-1, 0.5, 3, np.inf, np.nan])

    def test_ppf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])

        # Regular checks
        ## Outputs for q=0 and q=1 should be equal to minimum and maximum elements
        q = np.array([0, 0.1 - h, 0.1, 0.1 + h, 0.3 - h, 0.3, 0.3 + h, 1 - h, 1])
        assert_array_equal(rv.ppf(q), np.array([0.5, 0.5, 0.5, 1, 1, 1, 3, 3, 3]))

        # Coercion of not ndarray input
        _test_input_coercion(rv.ppf, q)

        # Bad input
        q = np.array([-np.inf, -h, np.nan, 1 + h, np.inf])
        assert_array_equal(
            rv.ppf(q), np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )

        # Broadcasting
        q = np.array([[0, 0.5], [0.0, 1.0]])
        assert_array_equal(rv.ppf(q), np.array([[0.5, 3], [0.5, 3]]))

        # One value input
        _test_one_value_input(rv.ppf, 0.25)
        _test_one_value_input(rv.ppf, -1)
        _test_one_value_input(rv.ppf, np.nan)

    def test_isf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        q_ref = np.array([0, 0.7 - h, 0.7, 0.7 + h, 0.9 - h, 0.9, 0.9 + h, 1 - h, 1])

        # Output is lowest value `t` (within supp.) for which `P(X > t) <= q`
        assert_array_equal(rv.isf(q_ref), rv.ppf(1 - q_ref))

    def test_rvs(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])

        _test_rvs_method(rv)

    def test__cdf_spline(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        x = np.array([-10, 0.5 - h, 0.5, 0.5 + h, 1 - h, 1, 1 + h, 3 - h, 3, 3 + h, 10])
        assert_array_equal(rv._cdf_spline(x), rv.cdf(x))

    def test_integrate_cdf(self):
        rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
        assert np.allclose(rv.integrate_cdf(-10, 10), 0.1 * 0.5 + 0.3 * 2 + 1 * 7)

    def test_convert(self):
        import randomvars._boolean as bool
        import randomvars._continuous as cont
        import randomvars._mixture as mixt

        rv = Disc([0.0, 0.5, 1, 3], [0.05, 0.05, 0.2, 0.7])

        # By default and supplying `None` should return self
        assert rv.convert() is rv
        assert rv.convert(None) is rv

        # Converting to Bool should result into boolean with probability of
        # `False` being probability of 0
        out_bool = rv.convert("Bool")
        assert isinstance(out_bool, bool.Bool)
        assert_array_almost_equal(out_bool.prob_false, rv.pmf(0.0), decimal=DECIMAL)

        # Converting to Cont should result into continuous RV with the same `x`
        # values as in input's xp-grid
        out_cont = rv.convert("Cont")
        assert isinstance(out_cont, cont.Cont)
        assert_array_equal(out_cont.x, rv.x)

        ## Converting to Cont if there is one element should raise an error
        with pytest.raises(ValueError, match="one"):
            Disc([1], [1]).convert("Cont")

        # Converting to own class should return self
        out_disc = rv.convert("Disc")
        assert out_disc is rv

        # Converting to Mixt should result into degenerate mixture with only
        # discrete component
        out_mixt = rv.convert("Mixt")
        assert isinstance(out_mixt, mixt.Mixt)
        assert out_mixt.disc is rv
        assert out_mixt.weight_disc == 1.0

        # Any other target class should result into error
        with pytest.raises(ValueError, match="one of"):
            rv.convert("aaa")


class TestFromRVAccuracy:
    """Accuracy of `Disc.from_rv()`"""

    def test_tails(self):
        test_passed = {
            name: TestFromRVAccuracy.is_from_rv_small_tails(distr)
            for name, distr in DISTRIBUTIONS.items()
        }

        assert all(test_passed.values())

    def test_small_prob_detection(self):
        # Currently small probabilities can be not detected due to "stepping"
        # procedure of x-values detection.

        # If small probability element is not detected, its probability is
        # "squashed" to the next (bigger) detected element
        rv_1 = distrs.rv_discrete(values=([1, 2, 3], [0.5, 0.125, 0.375]))
        with config.context({"small_prob": 0.125 + 1e-5}):
            rv_1_out = Disc.from_rv(rv_1)
            rv_1_ref = Disc([1, 3], [0.5, 0.5])
            _test_equal_rand(rv_1_out, rv_1_ref)

        # Currently not all elements with small probabilities are not detected,
        # but only those, which cumulative probability after previous detected
        # element is less than threshold.
        rv_2 = distrs.rv_discrete(values=([1, 2, 3, 4], [0.5, 0.0625, 0.0625, 0.375]))
        with config.context({"small_prob": 0.1}):
            rv_2_out = Disc.from_rv(rv_2)
            rv_2_ref = Disc([1, 3, 4], [0.5, 0.125, 0.375])
            _test_equal_rand(rv_2_out, rv_2_ref)

    def test_last_value(self):
        # If last (biggest) value has probability equal to `small_prob`, it
        # should nevertheless be included
        rv = distrs.rv_discrete(values=([0, 1], [0.875, 0.125]))
        with config.context({"small_prob": 0.125}):
            rv_out = Disc.from_rv(rv)
            rv_ref = Disc([0, 1], [0.875, 0.125])
            _test_equal_rand(rv_out, rv_ref)

        # If last value cuts some right tail, all its probability should be
        # transferred into last x-value
        rv = distrs.rv_discrete(values=([0, 1, 2, 3], [0.625, 0.125, 0.125, 0.125]))
        with config.context({"small_prob": 0.2}):
            rv_out = Disc.from_rv(rv)
            rv_ref = Disc([0, 2], [0.625, 0.375])
            _test_equal_rand(rv_out, rv_ref)

    @staticmethod
    def is_from_rv_small_tails(rv):
        small_prob = config.small_prob

        rv_disc = Disc.from_rv(rv)
        x = rv_disc.x

        # Left tail
        left_tail_prob = rv.cdf(x[0] - 1e-13)

        # Right tail
        right_tail_prob = 1 - rv.cdf(x[-1])

        return (left_tail_prob <= small_prob) and (right_tail_prob <= small_prob)
