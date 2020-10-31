# pylint: disable=missing-function-docstring
"""Tests for '_continuous.py' file"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.stats.distributions as distrs
from scipy.integrate import quad
import pytest

from randomvars._continuous import Cont
from randomvars._utils import _assert_equal_seq
from randomvars.options import get_option, option_context


DISTRIBUTIONS_COMMON = {
    "beta": distrs.beta(a=10, b=20),
    "chi_sq": distrs.chi2(df=10),
    "expon": distrs.expon(),
    "f": distrs.f(dfn=20, dfd=20),
    "gamma": distrs.gamma(a=10),
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


def assert_equal_cont(rv_1, rv_2):
    grid_1 = rv_1.x, rv_1.y
    grid_2 = rv_2.x, rv_2.y
    _assert_equal_seq(grid_1, grid_2)


def assert_almost_equal_cont(rv_1, rv_2, decimal=10):
    assert_array_almost_equal(rv_1.x, rv_2.x, decimal=decimal)
    assert_array_almost_equal(rv_1.y, rv_2.y, decimal=decimal)


def from_sample_max_error(x):
    rv = Cont.from_sample(x)
    density = get_option("density_estimator")(x)

    return np.max(np.abs(density(rv.x) - rv.y))


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
            with pytest.raises(ValueError, match=f"`{var}`.*numpy array"):
                def_args[var] = {"a": None}
                Cont(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*numeric"):
                def_args[var] = ["a", "a"]
                Cont(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*finite values"):
                def_args[var] = [0, np.nan]
                Cont(**def_args)
            with pytest.raises(ValueError, match=f"`{var}`.*finite values"):
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
            assert_equal_cont(rv, rv_ref)

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
        assert_equal_cont(rv_1, rv_ref)

        # Check if `y` is normalized
        rv_2 = Cont(x=x_ref, y=10 * y_ref)
        assert_equal_cont(rv_2, rv_ref)

        # Check if `x` and `y` are rearranged if not sorted
        with pytest.warns(UserWarning, match="`x`.*not sorted"):
            rv_3 = Cont(x=x_ref[[1, 0, 2]], y=y_ref[[1, 0, 2]])
            assert_equal_cont(rv_3, rv_ref)

        # Check if duplicated values are removed from `x`
        with pytest.warns(UserWarning, match="duplicated"):
            # First pair of xy-grid is taken among duplicates
            rv_4 = Cont(x=x_ref[[0, 1, 1, 2]], y=y_ref[[0, 1, 2, 2]])
            assert_equal_cont(rv_4, rv_ref)

    def test_str(self):
        rv = Cont([0, 2, 4], [0, 1, 0])
        assert str(rv) == "Continuous RV with 2 intervals (support: [0.0, 4.0])"

        # Uses singular noun with one interval
        rv = Cont([0, 1], [1, 1])
        assert str(rv) == "Continuous RV with 1 interval (support: [0.0, 1.0])"

    def test_properties(self):
        """Tests for properties"""
        x = np.arange(11)
        y = np.repeat(0.1, 11)
        rv = Cont(x, y)

        assert_array_equal(rv.x, x)
        assert_array_equal(rv.y, y)
        assert rv.a == 0.0
        assert rv.b == 10.0

    def test_support(self):
        rv = Cont([0.5, 1.5, 4.5], [0, 0.5, 0])
        assert rv.support() == (0.5, 4.5)

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

        _assert_equal_seq(
            rv.pdf_coeffs(x),
            (np.array([0, 0, 0, 2, 2, 2, 0]), np.array([0, 1, 1, -1, -1, -1, 0])),
        )
        _assert_equal_seq(
            rv.pdf_coeffs(x, side="left"),
            (np.array([0, 0, 0, 0, 2, 2, 0]), np.array([0, 1, 1, 1, -1, -1, 0])),
        )
        _assert_equal_seq(
            rv.pdf_coeffs(np.array([-np.inf, np.nan, np.inf])),
            (np.array([0, np.nan, 0]), np.array([0, np.nan, 0])),
        )

    def test_from_rv_basic(self):
        uniform = distrs.uniform
        norm = distrs.norm

        # Basic usage
        rv_unif = Cont.from_rv(uniform)
        rv_unif_test = Cont(x=[0, 1], y=[1, 1])
        assert_almost_equal_cont(rv_unif, rv_unif_test, decimal=12)

        # Object of `Cont` class should be returned untouched
        rv = Cont.from_rv(uniform)
        rv.aaa = "Extra method"
        rv2 = Cont.from_rv(rv)
        assert_equal_cont(rv, rv2)
        assert "aaa" in dir(rv2)

        # Forced support edges
        rv_right = Cont.from_rv(uniform, supp=(0.5, None))
        rv_right_test = Cont([0.5, 1], [2, 2])
        assert_almost_equal_cont(rv_right, rv_right_test, decimal=12)

        rv_left = Cont.from_rv(uniform, supp=(None, 0.5))
        rv_left_test = Cont([0, 0.5], [2, 2])
        assert_almost_equal_cont(rv_left, rv_left_test, decimal=12)

        rv_mid = Cont.from_rv(uniform, supp=(0.25, 0.75))
        rv_mid_test = Cont([0.25, 0.75], [2, 2])
        assert_almost_equal_cont(rv_mid, rv_mid_test, decimal=12)

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
        with option_context({"small_prob": 1e-6}):
            rv_norm = Cont.from_rv(norm)
            assert_array_almost_equal(rv_norm.support(), norm.ppf([1e-6, 1 - 1e-6]))

        with option_context({"small_prob": 1e-6}):
            rv_norm_right = Cont.from_rv(norm, supp=(-1, None))
            assert_array_almost_equal(rv_norm_right.support(), [-1, norm.ppf(1 - 1e-6)])

        with option_context({"small_prob": 1e-6}):
            rv_norm_left = Cont.from_rv(norm, supp=(None, 1))
            assert_array_almost_equal(rv_norm_left.support(), [norm.ppf(1e-6), 1])

        # Usage of `n_grid` option
        with option_context({"n_grid": 11}):
            rv_norm_small = Cont.from_rv(norm)
        assert len(rv_norm_small.x) <= 20

        # Usage of `integr_tol` option
        with option_context({"integr_tol": 1e-4}):
            rv_norm_1 = Cont.from_rv(norm)
        with option_context({"integr_tol": 1e-1}):
            rv_norm_2 = Cont.from_rv(norm)
        ## Increasing tolerance should lead to decrease of density grid
        assert len(rv_norm_1.x) > len(rv_norm_2.x)

    def test_from_sample_basic(self):
        norm = distrs.norm()

        rng = np.random.default_rng(101)
        x = norm.rvs(100, random_state=rng)
        rv = Cont.from_sample(x)
        assert isinstance(rv, Cont)

    def test_from_sample_errors(self):
        with pytest.raises(ValueError, match="numeric numpy array"):
            Cont.from_sample(["a"])

        with pytest.raises(ValueError, match="1d"):
            Cont.from_sample([[1], [2]])

    def test_from_sample_options(self):
        norm = distrs.norm()

        rng = np.random.default_rng(101)
        x = norm.rvs(100, random_state=rng)

        # "density_estimator"
        def uniform_estimator(x):
            x_min, x_max = x.min(), x.max()

            def res(x):
                return np.where((x >= x_min) & (x <= x_max), 1 / (x_max - x_min), 0)

            return res

        with option_context({"density_estimator": uniform_estimator}):
            rv = Cont.from_sample(x)
        assert len(np.unique(rv.y)) == 1

        # "density_estimator" which returns allowed classes
        ## `Cont` object should be returned untouched
        rv_estimation = Cont([0, 1], [1, 1])
        rv_estimation.aaa = "Extra method"
        with option_context({"density_estimator": lambda x: rv_estimation}):
            rv = Cont.from_sample(np.asarray([0, 1, 2]))
            assert "aaa" in dir(rv)

        ## "Scipy" distribution should be forwarded to `Cont.from_rv()`
        rv_norm = distrs.norm()
        with option_context({"density_estimator": lambda x: rv_norm}):
            rv = Cont.from_sample(np.asarray([0, 1, 2]))
            rv_ref = Cont.from_rv(rv_norm)
            assert_equal_cont(rv, rv_ref)

        # "density_mincoverage"
        with option_context({"density_mincoverage": 0}):
            rv = Cont.from_sample(x)
        ## With minimal density mincoverage output range should be equal to
        ## sample range
        assert_array_equal(rv.x[[0, -1]], [x.min(), x.max()])

        # "n_grid"
        with option_context({"n_grid": 11}):
            rv = Cont.from_sample(x)
        assert len(rv.x) <= 22

        # "integr_tol"
        with option_context({"integr_tol": 1e5}):
            rv = Cont.from_sample(x)
        ## With very high integral tolerance downgridding should result into
        ## grid with two elements
        assert len(rv.x) == 2

    def test_from_sample_single_value(self):
        """How well `from_sample()` handles single unique value in sample

        Main problem here is how density range is initialized during estimation.
        """

        zero_vec = np.zeros(10)

        # Default density estimator can't handle situation with single unique
        # sample value (gives `LinAlgError: singular matrix`).

        # Case when sample width is zero but density is not zero
        density_centered_interval = make_circ_density([(-1, 1)])
        with option_context({"density_estimator": lambda x: density_centered_interval}):
            assert from_sample_max_error(zero_vec) <= 1e-4

        # Case when both sample width and density are zero
        density_shifted_interval = make_circ_density([(10, 20)])
        with option_context({"density_estimator": lambda x: density_shifted_interval}):
            assert from_sample_max_error(zero_vec) <= 1e-4

    def test_pdf(self):
        """Tests for `.pdf()` method"""
        rv = Cont([0, 1, 3], [0.5, 0.5, 0])

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
        rv_dirac = Cont([10 - 1e-8, 10, 10 + 1e-8], [0, 1, 0])
        x = np.array([10 - 1e-8, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + 1e-8])
        ## Accuracy is of order of 10 due to extreme magnitudes of values
        assert_array_almost_equal(
            rv_dirac.pdf(x), np.array([0, 0.5e8, 1e8, 0.5e8, 0]), decimal=-1
        )

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_equal(rv.pdf(x), np.array([[0.0, 0.5], [0.25, 0.0]]))

    def test_cdf(self):
        """Tests for `.cdf()` method"""
        rv_1 = Cont([0, 1, 2], [0, 1, 0])

        # Regular checks
        x = np.array([-1, 0, 0.5, 1, 1.5, 2, 3])
        assert_array_equal(rv_1.cdf(x), np.array([0, 0, 0.125, 0.5, 0.875, 1, 1]))

        # Bad input
        x = np.array([-np.inf, np.nan, np.inf])
        assert_array_equal(rv_1.cdf(x), np.array([0, np.nan, 1]))

        # Dirac-like random variable
        rv_dirac = Cont([10 - 1e-8, 10, 10 + 1e-8], [0, 1, 0])
        x = np.array([10 - 1e-8, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + 1e-8])
        assert_array_almost_equal(
            rv_dirac.cdf(x), np.array([0, 0.125, 0.5, 0.875, 1]), decimal=7
        )

        # Broadcasting
        x = np.array([[-1, 0.5], [2, 4]])
        assert_array_equal(rv_1.cdf(x), np.array([[0.0, 0.125], [1.0, 1.0]]))

    def test_ppf(self):
        """Tests for `.ppf()` method"""
        # `ppf()` method should be inverse to `cdf()` for every sensible input
        rv_1 = Cont([0, 1, 2], [0, 1, 0])

        # Regular checks
        q = np.array([0, 0.125, 0.5, 0.875, 1])
        assert_array_equal(rv_1.ppf(q), np.array([0, 0.5, 1, 1.5, 2]))

        # Bad input
        q = np.array([-np.inf, -1e-8, np.nan, 1 + 1e-8, np.inf])
        assert_array_equal(
            rv_1.ppf(q), np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        )

        # Dirac-like random variable
        rv_dirac = Cont([10 - 1e-8, 10, 10 + 1e-8], [0, 1, 0])
        q = np.array([0, 0.125, 0.5, 0.875, 1])
        assert_array_almost_equal(
            rv_dirac.ppf(q),
            np.array([10 - 1e-8, 10 - 0.5e-8, 10, 10 + 0.5e-8, 10 + 1e-8]),
            decimal=9,
        )

        # Broadcasting
        q = np.array([[0, 0.5], [0.0, 1.0]])
        assert_array_equal(rv_1.ppf(q), np.array([[0.0, 1.0], [0.0, 2.0]]))

    def test_rvs(self):
        """Tests for `.rvs()`"""
        rv_1 = Cont([0, 1, 2], [0, 1, 0])

        # Regular checks
        smpl = rv_1.rvs(size=10)
        assert np.all((rv_1.a <= smpl) & (smpl <= rv_1.b))

        # Treats default `size` as 1
        assert rv_1.rvs().shape == tuple()

        # Broadcasting
        smpl_array = rv_1.rvs(size=(10, 2))
        assert smpl_array.shape == (10, 2)

        # Usage of `random_state`
        smpl_1 = rv_1.rvs(size=100, random_state=np.random.RandomState(101))
        smpl_2 = rv_1.rvs(size=100, random_state=np.random.RandomState(101))
        assert_array_equal(smpl_1, smpl_2)

        # Usage of integer `random_state` as a seed
        smpl_1 = rv_1.rvs(size=100, random_state=101)
        smpl_2 = rv_1.rvs(size=100, random_state=101)
        assert_array_equal(smpl_1, smpl_2)


class TestFromRVAccuracy:
    """Accuracy of `Cont.from_rv()`"""

    # Output of `from_rv()` should have CDF that differs from original CDF by
    # no more than `thres`
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "distr_dict,thres",
        [
            (DISTRIBUTIONS_COMMON, 1e-4),
            (DISTRIBUTIONS_INF_DENSITY, 1e-2),
            (DISTRIBUTIONS_HEAVY_TAILS, 1e-2),
        ],
    )
    def test_cdf_maxerror(self, distr_dict, thres):
        test_passed = {
            name: TestFromRVAccuracy.from_rv_cdf_maxerror(distr) <= thres
            for name, distr in distr_dict.items()
        }

        assert all(test_passed.values())

    @staticmethod
    def from_rv_cdf_maxerror(rv_base, n_inner_points=10, **kwargs):
        rv_test = Cont.from_rv(rv_base, **kwargs)
        x_grid = TestFromRVAccuracy.augment_grid(rv_test.x, n_inner_points)
        err = rv_base.cdf(x_grid) - rv_test.cdf(x_grid)
        return np.max(np.abs(err))

    @staticmethod
    def augment_grid(x, n_inner_points):
        test_arr = [
            np.linspace(x[i], x[i + 1], n_inner_points + 1, endpoint=False)
            for i in np.arange(len(x) - 1)
        ]
        test_arr.append([x[-1]])
        return np.concatenate(test_arr)


class TestFromSampleAccuracy:
    """Accuracy of `Cont.from_sample()`"""

    # Output of `from_sample()` should differ from original density estimate by
    # no more than `thres` (with default density estimator)
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "distr_dict,thres",
        [
            (DISTRIBUTIONS_COMMON, 1e-3),
            (DISTRIBUTIONS_INF_DENSITY, 1e-3),
            (DISTRIBUTIONS_HEAVY_TAILS, 1e-3),
        ],
    )
    def test_close_densities(self, distr_dict, thres):
        rng = np.random.default_rng(101)
        test_passed = {
            name: TestFromSampleAccuracy.simulated_density_error(distr, rng) <= thres
            for name, distr in distr_dict.items()
        }

        assert all(test_passed.values())

    @pytest.mark.slow
    def test_density_range(self):
        density_estimator = get_option("density_estimator")
        density_mincoverage = get_option("density_mincoverage")
        rng = np.random.default_rng(101)

        def generate_density_coverage(distr):
            x = distr.rvs(size=100, random_state=rng)
            density = density_estimator(x)
            rv = Cont.from_sample(x)
            return quad(density, rv.x[0], rv.x[-1])[0]

        test_passed = {
            distr_name: generate_density_coverage(distr) >= density_mincoverage
            for distr_name, distr in DISTRIBUTIONS.items()
        }

        assert all(test_passed.values())

    @staticmethod
    def simulated_density_error(distr, rng):
        x = distr.rvs(size=100, random_state=rng)
        return from_sample_max_error(x)


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

    with option_context({"density_estimator": extra_estimator}):
        rv = Cont.from_sample(x)

    assert (rv.x[0] <= x.min()) and (rv.x[-1] >= x.max())
