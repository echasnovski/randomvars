# pylint: disable=missing-function-docstring
"""Tests for '_continuous.py' file"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.stats.distributions as distrs
from scipy.stats.kde import gaussian_kde
from scipy.integrate import quad
import pytest

from randomvars._continuous import Cont
from randomvars.tests.commontests import (
    DECIMAL,
    _test_equal_rand,
    _test_equal_seq,
    _test_from_rv_rand,
    _test_from_sample_rand,
    _test_input_coercion,
    _test_log_fun,
    _test_one_value_input,
    _test_rvs_method,
    declass,
    h,
)
from randomvars.options import config


DISTRIBUTIONS_COMMON = {
    "beta": distrs.beta(a=10, b=20),
    "chi_sq": distrs.chi2(df=10),
    "expon": distrs.expon(),
    "f": distrs.f(dfn=20, dfd=20),
    "gamma": distrs.gamma(a=10),
    "laplace": distrs.laplace(),
    "lognorm": distrs.lognorm(s=0.5),
    "norm": distrs.norm(),
    "norm2": distrs.norm(loc=10),
    "norm3": distrs.norm(scale=0.1),
    "norm4": distrs.norm(scale=10),
    "norm5": distrs.norm(loc=10, scale=0.1),
    "t": distrs.t(df=10),
    "uniform": distrs.uniform(),
    "uniform2": distrs.uniform(loc=10, scale=0.1),
    "weibull_max": distrs.weibull_max(c=2),
    "weibull_min": distrs.weibull_min(c=2),
}

DISTRIBUTIONS_INF_DENSITY = {
    "inf_beta_both": distrs.beta(a=0.4, b=0.6),
    "inf_beta_left": distrs.beta(a=0.5, b=2),
    "inf_beta_right": distrs.beta(a=2, b=0.5),
    "inf_chi_sq": distrs.chi2(df=1),
    "inf_weibull_max": distrs.weibull_max(c=0.5),
    "inf_weibull_min": distrs.weibull_min(c=0.5),
}

DISTRIBUTIONS_HEAVY_TAILS = {
    "heavy_cauchy": distrs.cauchy(),
    "heavy_lognorm": distrs.lognorm(s=1),
    "heavy_t": distrs.t(df=2),
}

DISTRIBUTIONS = {
    **DISTRIBUTIONS_COMMON,
    **DISTRIBUTIONS_HEAVY_TAILS,
    **DISTRIBUTIONS_INF_DENSITY,
}


def augment_grid(x, n_inner_points):
    test_arr = [
        np.linspace(x[i], x[i + 1], n_inner_points + 1, endpoint=False)
        for i in np.arange(len(x) - 1)
    ]
    test_arr.append([x[-1]])
    return np.concatenate(test_arr)


def from_sample_cdf_max_error(x):
    rv = Cont.from_sample(x)
    density = config.estimator_cont(x)

    x_grid = augment_grid(rv.x, 10)

    # Efficient way of computing `quad(density, -np.inf, x_grid)`
    x_grid_ext = np.concatenate([[-np.inf], x_grid])
    cdf_intervals = np.array(
        [
            quad(density, x_l, x_r)[0]
            for x_l, x_r in zip(x_grid_ext[:-1], x_grid_ext[1:])
        ]
    )
    cdf_grid = np.cumsum(cdf_intervals)

    err = cdf_grid - rv.cdf(x_grid)
    return np.max(np.abs(err))


def circle_fun(x, low, high):
    x = np.array(x)
    center = 0.5 * (high + low)
    radius = 0.5 * (high - low)

    res = np.zeros_like(x)

    center_dist = np.abs(x - center)
    is_in = center_dist <= radius
    res[is_in] = np.sqrt(radius ** 2 - center_dist[is_in] ** 2)

    return res


def make_circ_density(intervals):
    """Construct circular density

    Density looks like half-circles with diameters lying in elements of
    `intervals`. Total integral is equal to 1.

    Parameters
    ----------
    intervals : iterable with elements being 2-element iterables
        Iterable of intervals with non-zero density.

    Returns
    -------
    density : callable
        Function which returns density values.
    """

    def density(x):
        res = np.zeros_like(x)
        tot_integral = 0
        for low, high in intervals:
            res += circle_fun(x, low, high)
            # There is only half of circle
            tot_integral += np.pi * (high - low) ** 2 / 8

        return res / tot_integral

    return density


class TestCont:
    """Regression tests for `Cont` class"""

    def test_init_errors(self):
        def check_one_input(def_args, var):
            with pytest.raises(TypeError, match=f"`{var}`.*numpy array"):
                def_args[var] = {"a": None}
                Cont(**def_args)
            with pytest.raises(TypeError, match=f"`{var}`.*float"):
                def_args[var] = ["a", "a"]
                Cont(**def_args)
            with pytest.raises(TypeError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.nan]
                Cont(**def_args)
            with pytest.raises(TypeError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.inf]
                Cont(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*1d array"):
                def_args[var] = [[0, 1]]
                Cont(**def_args)

        check_one_input({"y": [1, 1]}, "x")
        check_one_input({"x": [0, 1]}, "y")

        with pytest.raises(ValueError, match="[Ll]engths.*match"):
            Cont([0, 1], [1, 1, 1])

        with pytest.raises(ValueError, match="two"):
            Cont([1], [1])

        with pytest.warns(UserWarning, match="`x`.*not sorted.*`x` and `y`"):
            rv = Cont([1, 0], [0, 2])
            rv_ref = Cont([0, 1], [2, 0])
            _test_equal_rand(rv, rv_ref)

        with pytest.raises(ValueError, match="`y`.*negative"):
            Cont([0, 1], [1, -1])

        with pytest.raises(ValueError, match="`y`.*no positive"):
            Cont([0, 1], [0, 0])

    def test_init(self):
        x_ref = np.array([0, 1, 2])
        y_ref = np.array([0, 1, 0])
        rv_ref = Cont(x_ref, y_ref)

        # Simple case with non-numpy input
        rv_1 = Cont(x=x_ref.tolist(), y=y_ref.tolist())
        _test_equal_rand(rv_1, rv_ref)

        # Check if `y` is normalized
        rv_2 = Cont(x=x_ref, y=10 * y_ref)
        _test_equal_rand(rv_2, rv_ref)

        # Check if `x` and `y` are rearranged if not sorted
        with pytest.warns(UserWarning, match="`x`.*not sorted"):
            rv_3 = Cont(x=x_ref[[1, 0, 2]], y=y_ref[[1, 0, 2]])
            _test_equal_rand(rv_3, rv_ref)

        # Check if duplicated values are removed from `x`
        with pytest.warns(UserWarning, match="duplicated"):
            # First pair of xy-grid is taken among duplicates
            rv_4 = Cont(x=x_ref[[0, 1, 1, 2]], y=y_ref[[0, 1, 2, 2]])
            _test_equal_rand(rv_4, rv_ref)

    def test_str(self):
        rv = Cont([0, 2, 4], [0, 1, 0])
        assert str(rv) == "Continuous RV with 2 intervals (support: [0.0, 4.0])"

        # Uses singular noun with one interval
        rv = Cont([0, 1], [1, 1])
        assert str(rv) == "Continuous RV with 1 interval (support: [0.0, 1.0])"

    def test_properties(self):
        x = np.arange(11)
        y = np.repeat(0.1, 11)
        rv = Cont(x, y)

        assert list(rv.params.keys()) == ["x", "y"]
        assert_array_equal(rv.params["x"], x)
        assert_array_equal(rv.params["y"], y)

        assert_array_equal(rv.x, x)
        assert_array_equal(rv.y, y)
        assert rv.a == 0.0
        assert rv.b == 10.0

    def test_support(self):
        rv = Cont([0.5, 1.5, 4.5], [0, 0.5, 0])
        assert rv.support() == (0.5, 4.5)

    def test_compress(self):
        # Zero tails
        ## Left tail
        _test_equal_rand(
            Cont([0, 1, 2, 3], [0, 0, 0, 2]).compress(), Cont([2, 3], [0, 2])
        )
        _test_equal_rand(
            Cont([0, 1, 2, 3], [0, 0, 1, 0]).compress(), Cont([1, 2, 3], [0, 1, 0])
        )

        ## Right tail
        _test_equal_rand(
            Cont([0, 1, 2, 3], [2, 0, 0, 0]).compress(), Cont([0, 1], [2, 0])
        )
        _test_equal_rand(
            Cont([0, 1, 2, 3], [0, 1, 0, 0]).compress(), Cont([0, 1, 2], [0, 1, 0])
        )

        ## Both tails
        _test_equal_rand(
            Cont([0, 1, 2, 3, 4], [0, 0, 1, 0, 0]).compress(),
            Cont([1, 2, 3], [0, 1, 0]),
        )

        # Extra linearity
        ## Non-zero slope
        _test_equal_rand(
            Cont([0, 1, 2, 3, 4], [0.5, 0.25, 0, 0.25, 0.5]).compress(),
            Cont([0, 2, 4], [0.5, 0, 0.5]),
        )

        ## Zero slope, non-zero y
        _test_equal_rand(
            Cont([0, 1, 2], [0.5, 0.5, 0.5]).compress(), Cont([0, 2], [0.5, 0.5])
        )

        ## Zero slope, zero y, outside of tails
        _test_equal_rand(
            Cont([0, 1, 2, 3, 4], [1, 0, 0, 0, 1]).compress(),
            Cont([0, 1, 3, 4], [1, 0, 0, 1]),
        )

        # All features
        _test_equal_rand(
            Cont(np.arange(14), [0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 1, 0, 0]).compress(),
            Cont([2, 4, 6, 8, 10, 11, 12], [0, 2, 2, 0, 0, 1, 0]),
        )

        # If nothing to compress, self should be returned
        rv = Cont([0, 1], [1, 1])
        assert rv.compress() is rv

    def test_ground(self):
        w = config.small_width

        # Basic usage
        rv = Cont([0, 1], [1, 1])
        _test_equal_rand(
            rv.ground(), Cont([-w, 0, w, 1 - w, 1, 1 + w], [0, 0.5, 1, 1, 0.5, 0])
        )

        # Argument `direction`
        _test_equal_rand(
            rv.ground(direction="both"),
            Cont([-w, 0, w, 1 - w, 1, 1 + w], [0, 0.5, 1, 1, 0.5, 0]),
        )
        _test_equal_rand(
            rv.ground(direction="left"), Cont([-w, 0, w, 1], [0, 0.5, 1, 1])
        )
        _test_equal_rand(
            rv.ground(direction="right"), Cont([0, 1 - w, 1, 1 + w], [1, 1, 0.5, 0])
        )
        _test_equal_rand(rv.ground(direction="none"), rv)

        # Argument `w`
        w2 = 0.1
        _test_equal_rand(
            rv.ground(w=w2, direction="both"),
            Cont([-w2, 0, w2, 1 - w2, 1, 1 + w2], [0, 0.5, 1, 1, 0.5, 0]),
        )

        # Close neighbors
        rv2 = Cont([0, 0.25 * w, 0.5, 1 - 0.1 * w, 1], [1, 1, 1, 1, 1])
        rv2_grounded = rv2.ground(direction="both")
        ## Check that only outer points were added
        assert_array_equal(rv2_grounded.x[1:-1], rv2.x)
        ## Check that grounded actually happend
        assert_array_equal(rv2_grounded.y[[0, -1]], 0.0)
        ## Check that non-edge x-values havae same y-values
        assert_array_equal(rv2_grounded.pdf(rv2.x[1:-1]), rv2.pdf(rv2.x[1:-1]))

    def test_ground_options(self):
        rv = Cont([0, 1], [1, 1])
        with config.context({"small_width": 0.1}):
            w = config.small_width
            _test_equal_rand(
                rv.ground(), Cont([-w, 0, w, 1 - w, 1, 1 + w], [0, 0.5, 1, 1, 0.5, 0])
            )

    def test_ground_errors(self):
        rv = Cont([0, 1], [1, 1])
        with pytest.raises(ValueError, match="one of"):
            rv.ground(direction="aaa")

    def test__coeffs_by_ind(self):
        # All coefficients are returned if no `ind` is specified
        rv = Cont([0, 1, 2], [0, 1, 0])
        inter, slope = rv._coeffs_by_ind()
        assert_array_equal(inter, [0, 2])
        assert_array_equal(slope, [1, -1])

    def test__grid_by_ind(self):
        # All grid elements are returned if no `ind` is specified
        rv = Cont([0, 1, 2], [0, 1, 0])
        x_out, y_out, p_out = rv._grid_by_ind()
        x_ref, y_ref = rv.x, rv.y
        assert_array_equal(x_out, x_ref)
        assert_array_equal(y_out, y_ref)

    def test_pdf_coeffs(self):
        rv = Cont([0, 1, 2], [0, 1, 0])
        x = np.array([-1, 0, 0.5, 1, 1.5, 2, 2.5])

        with pytest.raises(ValueError, match="one of"):
            rv.pdf_coeffs(x, side="a")

        _test_equal_seq(
            rv.pdf_coeffs(x),
            (np.array([0, 0, 0, 2, 2, 2, 0]), np.array([0, 1, 1, -1, -1, -1, 0])),
        )
        _test_equal_seq(
            rv.pdf_coeffs(x, side="left"),
            (np.array([0, 0, 0, 0, 2, 2, 0]), np.array([0, 1, 1, 1, -1, -1, 0])),
        )
        _test_equal_seq(
            rv.pdf_coeffs(np.array([-np.inf, np.nan, np.inf])),
            (np.array([0, np.nan, 0]), np.array([0, np.nan, 0])),
        )

    def test_from_rv_basic(self):
        uniform = distrs.uniform
        norm = distrs.norm

        # Basic usage
        rv_unif = Cont.from_rv(uniform)
        rv_unif_test = Cont(x=[0, 1], y=[1, 1])
        _test_equal_rand(rv_unif, rv_unif_test, decimal=DECIMAL)

        # Objects of `Rand` class should be `convert()`ed
        _test_from_rv_rand(cls=Cont, to_class="Cont")

        # Forced support edges
        rv_right = Cont.from_rv(uniform, supp=(0.5, None))
        rv_right_test = Cont([0.5, 1], [2, 2])
        _test_equal_rand(rv_right, rv_right_test, decimal=DECIMAL)

        rv_left = Cont.from_rv(uniform, supp=(None, 0.5))
        rv_left_test = Cont([0, 0.5], [2, 2])
        _test_equal_rand(rv_left, rv_left_test, decimal=DECIMAL)

        rv_mid = Cont.from_rv(uniform, supp=(0.25, 0.75))
        rv_mid_test = Cont([0.25, 0.75], [2, 2])
        _test_equal_rand(rv_mid, rv_mid_test, decimal=DECIMAL)

    def test_from_rv_errors(self):
        # Absence of either `cdf` or `ppf` method should result intro error
        class Tmp:
            pass

        tmp1 = Tmp()
        tmp1.ppf = lambda x: np.where((0 <= x) & (x <= 1), 1, 0)
        with pytest.raises(ValueError, match="cdf"):
            Cont.from_rv(tmp1)

        tmp2 = Tmp()
        tmp2.cdf = lambda x: np.where((0 <= x) & (x <= 1), 1, 0)
        with pytest.raises(ValueError, match="ppf"):
            Cont.from_rv(tmp2)

    def test_from_rv_options(self):
        norm = distrs.norm

        # Finite support detection and usage of `small_prob` option
        with config.context({"small_prob": 1e-6}):
            rv_norm = Cont.from_rv(norm)
            assert_array_almost_equal(
                rv_norm.support(), norm.ppf([1e-6, 1 - 1e-6]), decimal=DECIMAL
            )

        with config.context({"small_prob": 1e-6}):
            rv_norm_right = Cont.from_rv(norm, supp=(-1, None))
            assert_array_almost_equal(
                rv_norm_right.support(), [-1, norm.ppf(1 - 1e-6)], decimal=DECIMAL
            )

        with config.context({"small_prob": 1e-6}):
            rv_norm_left = Cont.from_rv(norm, supp=(None, 1))
            assert_array_almost_equal(
                rv_norm_left.support(), [norm.ppf(1e-6), 1], decimal=DECIMAL
            )

        # Usage of `n_grid` option
        with config.context({"n_grid": 11}):
            rv_norm_small = Cont.from_rv(norm)
        assert len(rv_norm_small.x) <= 20

        # Usage of `cdf_tolerance` option
        with config.context({"cdf_tolerance": 1e-4}):
            rv_norm_1 = Cont.from_rv(norm)
        with config.context({"cdf_tolerance": 1e-1}):
            rv_norm_2 = Cont.from_rv(norm)
        ## Increasing CDF tolerance should lead to decrease of density grid
        assert len(rv_norm_1.x) > len(rv_norm_2.x)

    def test_from_sample_basic(self):
        norm = distrs.norm()

        rng = np.random.default_rng(101)
        x = norm.rvs(100, random_state=rng)
        rv = Cont.from_sample(x)
        assert isinstance(rv, Cont)

    def test_from_sample_errors(self):
        with pytest.raises(TypeError, match="numpy array with float"):
            Cont.from_sample(["a"])

        with pytest.raises(ValueError, match="1d"):
            Cont.from_sample([[1], [2]])

    def test_from_sample_options(self):
        norm = distrs.norm()

        rng = np.random.default_rng(101)
        x = norm.rvs(100, random_state=rng)

        # "estimator_cont"
        def uniform_estimator(x):
            x_min, x_max = x.min(), x.max()

            def res(x):
                return np.where((x >= x_min) & (x <= x_max), 1 / (x_max - x_min), 0)

            return res

        with config.context({"estimator_cont": uniform_estimator}):
            rv = Cont.from_sample(x)
        assert len(rv.y) == 2
        assert np.allclose(rv.y, rv.y[0], atol=1e-13)

        # "estimator_cont" which returns allowed classes
        ## `Rand` class should be forwarded to `from_rv()` method
        _test_from_sample_rand(
            cls=Cont,
            sample=x,
            estimator_option="estimator_cont",
        )

        ## "Scipy" distribution should be forwarded to `Cont.from_rv()`
        rv_norm = distrs.norm()
        with config.context({"estimator_cont": lambda x: rv_norm}):
            rv = Cont.from_sample(np.asarray([0, 1, 2]))
            rv_ref = Cont.from_rv(rv_norm)
            _test_equal_rand(rv, rv_ref)

        # "density_mincoverage"
        with config.context({"density_mincoverage": 0.0}):
            rv = Cont.from_sample(x)
        ## With minimal density mincoverage output range should be equal to
        ## sample range
        assert_array_equal(rv.x[[0, -1]], [x.min(), x.max()])

        # "n_grid"
        with config.context({"n_grid": 11}):
            rv = Cont.from_sample(x)
        assert len(rv.x) <= 22

        # "cdf_tolerance"
        with config.context({"cdf_tolerance": 2.0}):
            rv = Cont.from_sample(x)
        ## With very high CDF tolerance downgridding should result into grid
        ## with three elements. That is because CDF is approximated with
        ## simplest quadratic spline with single segment. That requires three
        ## knots.
        assert len(rv.x) == 3

    @pytest.mark.slow
    def test_from_sample_single_value(self):
        """How well `from_sample()` handles single unique value in sample

        Main problem here is how density range is initialized during estimation.
        """

        zero_vec = np.zeros(10)

        # Default density estimator can't handle situation with single unique
        # sample value (gives `LinAlgError: singular matrix`).

        # Case when sample width is zero but density is not zero
        density_centered_interval = make_circ_density([(-1, 1)])
        with config.context({"estimator_cont": lambda x: density_centered_interval}):
            assert from_sample_cdf_max_error(zero_vec) <= 1e-4

        # Case when both sample width and density are zero
        density_shifted_interval = make_circ_density([(10, 20)])
        with config.context({"estimator_cont": lambda x: density_shifted_interval}):
            # Here currently the problem is that support is estimated way to
            # wide with very small (~1e-9) non-zero density outside of [10,
            # 20]. However, CDFs are still close.
            assert from_sample_cdf_max_error(zero_vec) <= 2e-4

    def test_pdf(self):
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])

        # Regular checks
        x = np.array([-1, 0, 0.5, 1, 2, 3, 4])
        assert_array_equal(rv.pdf(x), np.array([0, 0.5, 0.5, 0.5, 0.25, 0, 0]))

        # Coercion of not ndarray input
        _test_input_coercion(rv.pdf, x)

        # Input around edges
        x = np.array([0 - 1e-10, 0 + 1e-10, 3 - 1e-10, 3 + 1e-10])
        assert_array_almost_equal(
            rv.pdf(x), np.array([0, 0.5, 0.25e-10, 0]), decimal=DECIMAL
        )

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv.pdf(x), np.array([0, np.nan, 0]))

        # Dirac-like random variable
        rv_dirac = Cont([10 - h, 10, 10 + h], [0, 1, 0])
        x = np.array([10 - h, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + h])
        ## Accuracy is of order of 10 due to extreme magnitudes of values
        assert_array_almost_equal(
            rv_dirac.pdf(x), np.array([0, 0.5e8, 1e8, 0.5e8, 0]), decimal=-1
        )

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_equal(rv.pdf(x), np.array([[0.0, 0.5], [0.25, 0.0]]))

        # One value input
        _test_one_value_input(rv.pdf, 0.5)
        _test_one_value_input(rv.pdf, -1)
        _test_one_value_input(rv.pdf, np.nan)

    def test_logpdf(self):
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])
        _test_log_fun(rv.logpdf, rv.pdf, x_ref=[-1, 0.1, 3, np.inf, np.nan])

    def test_pmf(self):
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])
        with pytest.raises(AttributeError, match=r"Use `pdf\(\)`"):
            rv.pmf(0)

    def test_logpmf(self):
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])
        with pytest.raises(AttributeError, match=r"Use `logpdf\(\)`"):
            rv.logpmf(0)

    def test_cdf(self):
        rv_1 = Cont([0, 1, 2], [0, 1, 0])

        # Regular checks
        x = np.array([-1, 0, 0.5, 1, 1.5, 2, 3])
        assert_array_equal(rv_1.cdf(x), np.array([0, 0, 0.125, 0.5, 0.875, 1, 1]))

        # Coercion of not ndarray input
        _test_input_coercion(rv_1.cdf, x)

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv_1.cdf(x), np.array([0, np.nan, 1]))

        # Dirac-like random variable
        rv_dirac = Cont([10 - h, 10, 10 + h], [0, 1, 0])
        x = np.array([10 - h, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + h])
        assert_array_almost_equal(
            rv_dirac.cdf(x), np.array([0, 0.125, 0.5, 0.875, 1]), decimal=DECIMAL
        )

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_equal(rv_1.cdf(x), np.array([[0.0, 0.125], [1.0, 1.0]]))

        # One value input
        _test_one_value_input(rv_1.cdf, 0.5)
        _test_one_value_input(rv_1.cdf, -1)
        _test_one_value_input(rv_1.cdf, np.nan)

    def test_logcdf(self):
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])
        _test_log_fun(rv.logcdf, rv.cdf, x_ref=[-1, 0.1, 3, np.inf, np.nan])

    def test_sf(self):
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])
        x_ref = [-1, 0.1, 3, np.inf, np.nan]
        assert_array_equal(rv.sf(x_ref), 1 - rv.cdf(x_ref))

    def test_logsf(self):
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])
        _test_log_fun(rv.logsf, rv.sf, x_ref=[-1, 0.1, 3, np.inf, np.nan])

    def test_ppf(self):
        # `ppf()` method should be inverse to `cdf()` for every sensible input
        rv_1 = Cont([0, 1, 2], [0, 1, 0])

        # Regular checks
        q = np.array([0, 0.125, 0.5, 0.875, 1])
        assert_array_equal(rv_1.ppf(q), np.array([0, 0.5, 1, 1.5, 2]))

        # Coercion of not ndarray input
        _test_input_coercion(rv_1.ppf, q)

        # Bad input
        q = np.array([-np.inf, -h, np.nan, 1 + h, np.inf])
        assert_array_equal(
            rv_1.ppf(q), np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )

        # Dirac-like random variable
        rv_dirac = Cont([10 - h, 10, 10 + h], [0, 1, 0])
        q = np.array([0, 0.125, 0.5, 0.875, 1])
        assert_array_almost_equal(
            rv_dirac.ppf(q),
            np.array([10 - h, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + h]),
            decimal=DECIMAL,
        )

        # Broadcasting
        q = np.array([[0, 0.5], [0.0, 1.0]])
        assert_array_equal(rv_1.ppf(q), np.array([[0.0, 1.0], [0.0, 2.0]]))

        # One value input
        _test_one_value_input(rv_1.ppf, 0.25)
        _test_one_value_input(rv_1.ppf, -1)
        _test_one_value_input(rv_1.ppf, np.nan)

        # Should return the smallest x-value in case of zero-density interval(s)
        rv_zero_density = Cont([0, 1, 2, 3, 4, 5, 6], [0, 0.5, 0, 0, 0, 0.5, 0])
        assert rv_zero_density.ppf(0.5) == 2

    def test_isf(self):
        rv = Cont([0, 1, 2], [0, 1, 0])

        # Regular checks
        q_ref = np.array([0, 0.125, 0.5, 0.875, 1])
        assert_array_equal(rv.sf(rv.isf(q_ref)), q_ref)

    def test_rvs(self):
        rv_1 = Cont([0, 1, 2], [0, 1, 0])

        _test_rvs_method(rv_1)

    def test__cdf_spline(self):
        rv = Cont([0, 1, 2], [0, 1, 0])
        x = [-10, 0, 0.5, 1, 1.5, 2, 10]
        assert_array_equal(rv._cdf_spline(x), rv.cdf(x))

    def test_integrate_cdf(self):
        rv = Cont([0, 1, 2], [0, 1, 0])
        assert np.allclose(rv.integrate_cdf(-10, 10), quad(rv.cdf, -10, 10)[0])

    def test_convert(self):
        import randomvars._boolean as bool
        import randomvars._discrete as disc
        import randomvars._mixture as mixt

        rv = Cont([0, 1, 2], [0, 1, 0])

        # By default and supplying `None` should return self
        assert rv.convert() is rv
        assert rv.convert(None) is rv

        # Converting to Bool should result into boolean with probability of
        # `False` being 0 (because probability of continuous RV being exactly
        # zero is 0).
        out_bool = rv.convert("Bool")
        assert isinstance(out_bool, bool.Bool)
        assert out_bool.prob_true == 1.0

        # Converting to own class should return self
        out_cont = rv.convert("Cont")
        assert out_cont is rv

        # Converting to Disc should result into discrete RV with the same `x`
        # values as in input's xy-grid
        out_disc = rv.convert("Disc")
        assert isinstance(out_disc, disc.Disc)
        assert_array_equal(out_disc.x, rv.x)

        # Converting to Mixt should result into degenerate mixture with only
        # continuous component
        out_mixt = rv.convert("Mixt")
        assert isinstance(out_mixt, mixt.Mixt)
        assert out_mixt.cont is rv
        assert out_mixt.weight_cont == 1.0

        # Any other target class should result into error
        with pytest.raises(ValueError, match="one of"):
            rv.convert("aaa")


class TestFromRVAccuracy:
    """Accuracy of `Cont.from_rv()`"""

    # Output of `from_rv()` should have CDF that differs from original CDF by
    # no more than `thres`
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "distr_dict,thres",
        [
            (DISTRIBUTIONS_COMMON, 1e-4),
            (DISTRIBUTIONS_INF_DENSITY, 1e-3),
            (DISTRIBUTIONS_HEAVY_TAILS, 5e-3),
        ],
    )
    def test_cdf_maxerror(self, distr_dict, thres):
        test_passed = {
            name: TestFromRVAccuracy.from_rv_cdf_maxerror(distr) <= thres
            for name, distr in distr_dict.items()
        }

        assert all(test_passed.values())

    def test_detected_support(self):
        """Test correct trimming of zero tails"""
        rv_ref = Cont([0, 1, 2, 3, 4], [0, 0, 1, 0, 0])
        rv_out = Cont.from_rv(declass(rv_ref))
        _test_equal_rand(rv_out, rv_ref.compress(), decimal=4)

    @staticmethod
    def from_rv_cdf_maxerror(rv_base, n_inner_points=10, **kwargs):
        rv_test = Cont.from_rv(rv_base, **kwargs)
        x_grid = augment_grid(rv_test.x, n_inner_points)
        err = rv_base.cdf(x_grid) - rv_test.cdf(x_grid)
        return np.max(np.abs(err))


class TestFromSampleAccuracy:
    """Accuracy of `Cont.from_sample()`"""

    # Output of `from_sample()` should differ from original density estimate by
    # no more than `thres` (with default density estimator)
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "distr_dict,thres",
        [
            (DISTRIBUTIONS_COMMON, 1e-4),
            (DISTRIBUTIONS_INF_DENSITY, 1.5e-4),
            (DISTRIBUTIONS_HEAVY_TAILS, 1e-4),
        ],
    )
    def test_close_cdf(self, distr_dict, thres):
        rng = np.random.default_rng(101)
        test_passed = {
            name: TestFromSampleAccuracy.simulated_cdf_error(distr, rng) <= thres
            for name, distr in distr_dict.items()
        }

        assert all(test_passed.values())

    @pytest.mark.slow
    def test_density_range(self):
        density_mincoverage = config.density_mincoverage
        estimator_cont = config.estimator_cont
        rng = np.random.default_rng(101)

        def generate_density_coverage(distr):
            x = distr.rvs(size=100, random_state=rng)
            density = estimator_cont(x)
            rv = Cont.from_sample(x)
            return quad(density, rv.x[0], rv.x[-1])[0]

        test_passed = {
            distr_name: generate_density_coverage(distr) >= density_mincoverage
            for distr_name, distr in DISTRIBUTIONS.items()
        }

        assert all(test_passed.values())

    @staticmethod
    def simulated_cdf_error(distr, rng):
        x = distr.rvs(size=100, random_state=rng)

        # Testing with `gaussian_kde` as the most used density estimator. This
        # also enables to use rather fast way of computing CDF of estimated
        # density via `integrate_box_1d` method.
        with config.context({"estimator_cont": gaussian_kde}):
            rv = Cont.from_sample(x)
            density = config.estimator_cont(x)

        x_grid = augment_grid(rv.x, 10)

        # Interestingly enough, direct computation with `-np.inf` as left
        # integration limit is both accurate and more efficient than computing
        # integrals for each segment and then use `np.cumsum()`. Probably this
        # is because integration of gaussian curves with infinite left limit is
        # done directly through gaussian CDF.
        cdf_grid = np.array(
            [density.integrate_box_1d(-np.inf, cur_x) for cur_x in x_grid]
        )

        err = cdf_grid - rv.cdf(x_grid)
        return np.max(np.abs(err))


def test__extend_range():
    def extra_estimator(x):
        x_min, x_max = x.min(), x.max()
        prob_height = 1 / (x_max - x_min + 1)

        def res(x):
            return np.where(
                ((x_min < x) & (x < x_max)) | ((x_max + 1 < x) & (x < x_max + 2)),
                prob_height,
                0,
            )

        return res

    norm = distrs.norm()
    rng = np.random.default_rng(101)
    x = norm.rvs(100, random_state=rng)

    with config.context({"estimator_cont": extra_estimator}):
        rv = Cont.from_sample(x)

    assert (rv.x[0] <= x.min()) and (rv.x[-1] >= x.max())
