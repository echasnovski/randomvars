# pylint: disable=missing-function-docstring
"""Tests for 'rv_piecelin.py' file
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.stats.distributions as distrs
import pytest

from randomvars.rv_piecelin import rv_piecelin


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
    "beta_both": distrs.beta(a=0.4, b=0.6),
    "beta_left": distrs.beta(a=0.5, b=2),
    "beta_right": distrs.beta(a=2, b=0.5),
    "chi_sq": distrs.chi2(df=1),
    "weibull_max": distrs.weibull_max(c=0.5),
    "weibull_min": distrs.weibull_min(c=0.5),
}

DISTRIBUTIONS_HEAVY_TAILS = {
    "cauchy": distrs.cauchy(),
    "lognorm": distrs.lognorm(s=1),
    "t": distrs.t(df=2),
}


def assert_equal_seq(first, second, *args, **kwargs):
    assert len(first) == len(second)
    for el1, el2 in zip(first, second):
        assert_array_equal(el1, el2, *args, **kwargs)


def assert_equal_rv_piecelin(rv_p_1, rv_p_2):
    grid_1 = rv_p_1.x, rv_p_1.y, rv_p_1.p
    grid_2 = rv_p_2.x, rv_p_2.y, rv_p_2.p
    assert_equal_seq(grid_1, grid_2)


def assert_almost_equal_rv_piecelin(rv_p_1, rv_p_2, decimal=10):
    assert_array_almost_equal(rv_p_1.x, rv_p_2.x, decimal=decimal)
    assert_array_almost_equal(rv_p_1.y, rv_p_2.y, decimal=decimal)
    assert_array_almost_equal(rv_p_1.p, rv_p_2.p, decimal=decimal)


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
            assert_equal_rv_piecelin(rv, rv_ref)

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
        assert_equal_rv_piecelin(rv_1, rv_ref)

        # Check if `y` is normalized
        rv_2 = rv_piecelin(x=x_ref, y=10 * y_ref)
        assert_equal_rv_piecelin(rv_2, rv_ref)

        # Check if `x` and `y` are rearranged if not sorted
        rv_3 = rv_piecelin(x=x_ref[[1, 0, 2]], y=10 * y_ref[[1, 0, 2]])
        assert_equal_rv_piecelin(rv_3, rv_ref)

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

    def test_from_rv(self):
        uniform = distrs.uniform
        norm = distrs.norm

        # Basic usage
        rv_unif = rv_piecelin.from_rv(uniform)
        rv_unif_test = rv_piecelin(x=[0, 1], y=[1, 1])
        assert_almost_equal_rv_piecelin(rv_unif, rv_unif_test, decimal=12)

        # Forced support edges
        rv_right = rv_piecelin.from_rv(uniform, supp=(0.5, None))
        rv_right_test = rv_piecelin([0.5, 1], [2, 2])
        assert_almost_equal_rv_piecelin(rv_right, rv_right_test, decimal=12)

        rv_left = rv_piecelin.from_rv(uniform, supp=(None, 0.5))
        rv_left_test = rv_piecelin([0, 0.5], [2, 2])
        assert_almost_equal_rv_piecelin(rv_left, rv_left_test, decimal=12)

        rv_mid = rv_piecelin.from_rv(uniform, supp=(0.25, 0.75))
        rv_mid_test = rv_piecelin([0.25, 0.75], [2, 2])
        assert_almost_equal_rv_piecelin(rv_mid, rv_mid_test, decimal=12)

        # Finite support detection
        rv_norm = rv_piecelin.from_rv(norm, tail_prob=1e-6)
        assert_array_almost_equal(rv_norm.support(), norm.ppf([1e-6, 1 - 1e-6]))

        rv_norm_right = rv_piecelin.from_rv(norm, supp=(-1, None), tail_prob=1e-6)
        assert_array_almost_equal(rv_norm_right.support(), [-1, norm.ppf(1 - 1e-6)])

        rv_norm_left = rv_piecelin.from_rv(norm, supp=(None, 1), tail_prob=1e-6)
        assert_array_almost_equal(rv_norm_left.support(), [norm.ppf(1e-6), 1])

        # Usage of `n_grid` argument
        rv_norm_small = rv_piecelin.from_rv(norm, n_grid=11)
        assert len(rv_norm_small.x) <= 20

        # Usage of `integr_tol` argument
        rv_norm_1 = rv_piecelin.from_rv(norm, integr_tol=1e-4)
        rv_norm_2 = rv_piecelin.from_rv(norm, integr_tol=1e-1)
        ## Increasing tolerance should lead to decrease of density grid
        assert len(rv_norm_1.x) > len(rv_norm_2.x)

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


class TestFromRVAccuracy:
    """Accuracy of `rv_piecelin.from_rv()`"""

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
        maxerrors = {
            name: TestFromRVAccuracy.from_rv_cdf_maxerror(distr)
            for name, distr in distr_dict.items()
        }
        test_passed = {name: err <= thres for name, err in maxerrors.items()}

        assert all(test_passed.values())

    @staticmethod
    def from_rv_cdf_maxerror(rv_base, n_inner_points=10, **kwargs):
        rv_test = rv_piecelin.from_rv(rv_base, **kwargs)
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
