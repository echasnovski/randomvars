import numpy as np
from numpy.testing import assert_array_equal
import pytest

from randomvars._utils import (
    _as_1d_numpy,
    _sort_parallel,
    _unique_parallel,
    _searchsorted_wrap,
    _find_nearest_ind,
    _copy_nan,
    _trapez_integral,
    _trapez_integral_cum,
    _quad_silent,
    _is_close,
    _is_zero,
    _tolerance,
    _minmax,
    _collapse_while_equal_fval,
    _assert_positive,
    BSplineConstExtrapolate,
)
from randomvars.tests.commontests import h
from randomvars.options import options


def test__as_1d_array():
    # Check for array-like
    with pytest.raises(TypeError, match=f"`tmp_name`.*numpy array"):
        _as_1d_numpy({"a": None}, "tmp_name")

    # Usage of `chkfinite` argument
    ## Should be `True` by default
    with pytest.raises(TypeError, match=f"`tmp_name`.*finite values"):
        _as_1d_numpy([0, np.nan], "tmp_name")
    with pytest.raises(TypeError, match=f"`tmp_name`.*finite values"):
        _as_1d_numpy([0, np.inf], "tmp_name")

    ## Shouldn't give errors if `False`
    _as_1d_numpy([0, np.nan], "tmp_name", chkfinite=False)
    _as_1d_numpy([0, np.inf], "tmp_name", chkfinite=False)

    ## Shouldn't mention finite values in error message if `False`
    with pytest.raises(TypeError, match=f"numeric numpy array.$"):
        _as_1d_numpy(["a"], "tmp_name", chkfinite=False)

    # Usage of `dtype` argument
    ## Should be "numeric" by default
    with pytest.raises(TypeError, match=f"`tmp_name`.*numeric"):
        _as_1d_numpy(["a", "a"], "tmp_name")
        _as_1d_numpy(["a", "a"], "tmp_name", dtype="float64")
    ## Also boolean dtype is accepted, but as every object in Python can be tested
    ## for being "truthy", it can't fail
    _as_1d_numpy([lambda x: x, {"a": 0}, np.inf], "tmp_name", dtype="bool")
    _as_1d_numpy(["a", "a"], "tmp_name", dtype="bool")

    # Check for 1d
    with pytest.raises(ValueError, match=f"`tmp_name`.*1d array"):
        _as_1d_numpy([[0, 1]], "tmp_name")


def test__sort_parallel():
    x = np.array([1, 0])
    y = np.array([1, 2])
    x_sorted = np.array([0, 1])
    y_sorted = np.array([2, 1])

    # Nothing is done if `x` is already sorted
    x_out, y_out = _sort_parallel(x_sorted, y_sorted)
    assert_array_equal(x_out, x_sorted)
    assert_array_equal(y_out, y_sorted)

    # Error is given if lengths differ
    with pytest.raises(ValueError, match="[Ll]engths of `x` and `tmp_name`"):
        _sort_parallel([0, 1], [0, 1, 2], y_name="tmp_name")

    # Warning should be given by default
    with pytest.warns(UserWarning, match="`x`.*not sorted.*`x` and `tmp_name`"):
        x_out, y_out = _sort_parallel(x, y, y_name="tmp_name")
        assert_array_equal(x_out, x_sorted)
        assert_array_equal(y_out, y_sorted)

    # No warning should be given if `warn=False`
    with pytest.warns(None) as record:
        x_out, y_out = _sort_parallel(x, y, warn=False)
        assert_array_equal(x_out, x_sorted)
        assert_array_equal(y_out, y_sorted)
    assert len(record) == 0


def test__unique_parallel():
    x = np.array([0, 1, 1, 2])
    y = np.array([0, 1, 2, 3])
    x_unique = np.array([0, 1, 2])
    y_unique = np.array([0, 1, 3])

    # Warning should be given by default
    with pytest.warns(UserWarning, match="duplicated values"):
        x_out, y_out = _unique_parallel(x, y)
        assert_array_equal(x_out, x_unique)
        assert_array_equal(y_out, y_unique)

    # No warning should be given if `warn=False`
    with pytest.warns(None) as record:
        x_out, y_out = _unique_parallel(x, y, warn=False)
        assert_array_equal(x_out, x_unique)
        assert_array_equal(y_out, y_unique)
    assert len(record) == 0


def test__searchsorted_wrap():
    # Normal usage
    out = _searchsorted_wrap([0, 1], [-np.inf, -1, 0, 0.5, 1, 2, np.inf, np.nan])
    out_ref = np.array([0, 0, 1, 1, 1, 2, 2, -1])
    assert_array_equal(out, out_ref)

    # Usage of `side` and `edge_inside` argument combinations
    assert_array_equal(
        _searchsorted_wrap([0, 1, 2], [0, 1, 2], side="left", edge_inside=False),
        np.array([0, 1, 2]),
    )
    assert_array_equal(
        _searchsorted_wrap([0, 1, 2], [0, 1, 2], side="left", edge_inside=True),
        np.array([1, 1, 2]),
    )
    assert_array_equal(
        _searchsorted_wrap([0, 1, 2], [0, 1, 2], side="right", edge_inside=False),
        np.array([1, 2, 3]),
    )
    assert_array_equal(
        _searchsorted_wrap([0, 1, 2], [0, 1, 2], side="right", edge_inside=True),
        np.array([1, 2, 2]),
    )

    # Index for `np.nan` input is `-1`
    assert _searchsorted_wrap([0, 1], [np.nan]) == -1


def test__find_nearest_ind():
    x = np.array([-10, -1.01, -1, -0.99, -0.5, -0.01, 0, 0.01, 0.5, 0.99, 1, 1.01, 10])
    v = np.array([-1, 0, 1])

    # Input errors
    with pytest.raises(ValueError, match="one"):
        _find_nearest_ind(x, v.reshape((1, -1)))

    with pytest.raises(ValueError, match="`side`.*one of"):
        _find_nearest_ind(x, v, side="aaa")

    # General usage
    out = _find_nearest_ind(x, v)
    out_ref = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    assert_array_equal(out, out_ref)

    out = _find_nearest_ind(x, v, side="right")
    out_ref = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    assert_array_equal(out, out_ref)

    # Unsorted reference array
    ord = np.array([1, 2, 0])
    ord_inv = np.array([2, 0, 1])
    v_unsorted = v[ord]

    out = _find_nearest_ind(x, v_unsorted)
    out_ref = ord_inv[_find_nearest_ind(x, v)]
    assert_array_equal(out, out_ref)

    # Broadcasting
    x_arr = np.array([[-2, 5, 0.5], [0.1, 0.7, 1.5]])
    out = _find_nearest_ind(x_arr, v)
    assert out.shape == x_arr.shape

    # `v` as scalar value
    out = _find_nearest_ind(x, 1)
    out_ref = np.repeat(0, len(x))
    assert_array_equal(out, out_ref)

    # `v` with single element
    out = _find_nearest_ind(x, [1])
    out_ref = np.repeat(0, len(x))
    assert_array_equal(out, out_ref)


def test__copy_nan():
    x = np.array([[np.nan, 1, 0], [np.inf, np.nan, np.nan]])
    y = np.zeros_like(x)
    out = _copy_nan(fr=x, to=y)
    out_ref = np.array([[np.nan, 0, 0], [0, np.nan, np.nan]])
    assert_array_equal(out, out_ref)
    # Check that `_copy_nan` returned copy of `from_arr` array
    assert_array_equal(y, np.zeros_like(x))


def test__trapez_integral():
    out = _trapez_integral(np.array([0, 1]), np.array([1, 1]))
    assert_array_equal(out, 1.0)

    out = _trapez_integral(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    assert_array_equal(out, 12.5)


def test__trapez_integral_cum():
    out = _trapez_integral_cum(np.array([0, 1]), np.array([1, 1]))
    assert_array_equal(out, np.array([0.0, 1.0]))

    out = _trapez_integral_cum(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    assert_array_equal(out, np.array([0.0, 5.0, 12.5]))


def test__quad_silent():
    # Normal usage
    out = _quad_silent(lambda x: x, 0, 1)
    assert out == 0.5

    # Here vanilla `quad()` gives `IntegrationWarning` about bad integrand
    # behavior, which should be suppressed by `quad_silent()`
    with pytest.warns(None) as record:
        _quad_silent(np.tan, 0, np.pi / 2.0 + 0.0001)
    assert len(record) == 0


def test__is_close():
    assert_array_equal(_is_close([1, 0, -1], [-1, 0, 1]), [False, True, False])

    with options.context({"base_tolerance": 0.1}):
        assert_array_equal(_is_close([0.15, 0.05], [0.0, 0.0]), [False, True])

    # Bad input
    bad_input = [1.0, np.nan, -np.inf, np.inf]
    assert_array_equal(_is_close(bad_input, 1), [True, False, False, False])
    assert_array_equal(_is_close(bad_input, np.nan), [False, False, False, False])
    assert_array_equal(_is_close(bad_input, -np.inf), [False, False, True, False])
    assert_array_equal(_is_close(bad_input, np.inf), [False, False, False, True])

    # Broadcasting
    assert_array_equal(_is_close([[1, 2]], [[1], [2]]), [[True, False], [False, True]])


def test__tolerance():
    with options.context({"base_tolerance": 0.1}):
        # Tolerance for values inside [-1, 1] should be base tolerance
        assert_array_equal(_tolerance([-1, 0.5, 1e-15, 0, 1e-15, 0.5, 1]), 0.1)

        # Tolerance for values outside [-1, 1] should be increased
        # proportionally to spacing
        cutoffs = 2 ** np.arange(10)
        assert_array_equal(_tolerance(cutoffs) / _tolerance(1), cutoffs)
        assert_array_equal(_tolerance(-cutoffs) / _tolerance(1), cutoffs)


def test__is_zero():
    assert_array_equal(_is_zero([1, 0, -1]), [False, True, False])

    with options.context({"base_tolerance": 0.1}):
        assert_array_equal(_is_zero([0.15, 0.05]), [False, True])


def test__minmax():
    assert _minmax([0, 9, 10, -1, np.nan]) == (-1, 10)
    assert _minmax([-np.inf, 10, np.inf, 20, np.nan]) == (-np.inf, np.inf)


def test__collapse_while_equal_fval():
    def assert_collapse(f, interval, side, reference):
        out = _collapse_while_equal_fval(f, interval, side)
        assert _is_close(out, reference)
        assert f(interval[side]) == f(out)

    # Usual cases
    assert_collapse(lambda x: 1 if x < -1 else 2, [-10, 1], 0, reference=-1)
    assert_collapse(lambda x: 1 if x < -1 else 2, [-10, 1], 1, reference=-1)

    # Case when function is constant on the whole input interval
    ## Input side should be dragged all the way to the other side and return it
    assert_collapse(lambda x: 1, [-10, 1], 0, reference=1)
    assert_collapse(lambda x: 1, [-10, 1], 1, reference=-10)

    # Case when function has equal values at input ends but is not constant on
    # the whole interval
    assert_collapse(lambda x: 1 if x < -1 else x, [-10, 1], 0, reference=-1)

    # Cases with no actual collapse
    assert_collapse(lambda x: x, [-1, 2], 0, reference=-1)
    assert_collapse(lambda x: x, [-1, 2], 1, reference=2)


def test__assert_positive():
    with pytest.raises(ValueError, match="`tmp_name`.*negative"):
        _assert_positive(np.array([-1, 0, 1]), x_name="tmp_name")
    with pytest.raises(ValueError, match="`tmp_name`.*no positive"):
        _assert_positive(np.array([0, 0, 0]), x_name="tmp_name")


class TestBSplineConstExtrapolate:
    def test_init(self):
        # Constant spline with values -0.5 in [-3, -2) and 0.5 in [-2, -1]
        # Extrapolates as -1 on (-inf, -3) and as 1 on (-1, inf)
        spline = BSplineConstExtrapolate(
            left=-1, right=1, t=[-3, -2, -1], c=[-0.5, 0.5], k=0
        )
        assert_array_equal(
            spline([-10, -3 - h, -3, -2.5, -2 - h, -2, -1.5, -1 - h, -1, -1 + h, 10]),
            np.array([-1, -1, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 1, 1, 1]),
        )
        assert_array_equal(spline([np.nan, -np.inf, np.inf]), [np.nan, -1, 1])

        # Linear spline inside [-3, -2] with one segment from (-3, -1) to (-2, 1).
        # Extrapolates as -2 in (-inf, -3) and 2 on (2, inf)
        spline = BSplineConstExtrapolate(
            left=-2, right=2, t=[-3, -3, -2, -2], c=[-1, 1, 0, 0], k=1
        )
        assert_array_equal(
            spline([-10, -3 - h, -3, -2.5, -2, -2 + h, 10]),
            np.array([-2, -2, -1, 0, 2, 2, 2]),
        )
        assert_array_equal(spline([np.nan, -np.inf, np.inf]), [np.nan, -2, 2])

        # Quadratic spline inside [-3, -2] (integration of linear spline with
        # one segment from (-3, -1) to (-2, 1)).
        # Extrapolates as -1 in (-inf, -3) and 2 in (-2, inf)
        spline = BSplineConstExtrapolate(
            left=-1, right=2, t=[-3, -3, -3, -2, -2, -2], c=[0, -0.5, 0, 0, 0, 0], k=2
        )
        assert_array_equal(
            spline([-10, -3 - h, -3, -2.5, -2, -2 + h, 10]),
            np.array([-1, -1, 0, -0.25, 2, 2, 2]),
        )
        assert_array_equal(spline([np.nan, -np.inf, np.inf]), [np.nan, -1, 2])

    def test_integrate(self):
        # Constant spline with values -0.5 in [-3, -2) and 0.5 in [-2, -1]
        # Extrapolates as -1 on (-inf, -3) and as 1 on (-1, inf)
        spline = BSplineConstExtrapolate(
            left=-1, right=1, t=[-3, -2, -1], c=[-0.5, 0.5], k=0
        )
        assert np.allclose(
            spline.integrate(-10, 10), -1 * 7 + (-0.5) * 1 + 0.5 * 1 + 1 * 11
        )
        assert np.allclose(
            spline.integrate(10, -10), -(-1 * 7 + (-0.5) * 1 + 0.5 * 1 + 1 * 11)
        )

        # Linear spline inside [-3, -2] with one segment from (-3, -1) to (-2, 1).
        # Extrapolates as -2 in (-inf, -3) and 2 on (2, inf)
        spline = BSplineConstExtrapolate(
            left=-2, right=2, t=[-3, -3, -2, -2], c=[-1, 1, 0, 0], k=1
        )
        assert np.allclose(spline.integrate(-10, 10), -2 * 7 + 0 + 2 * 12)
        assert np.allclose(spline.integrate(10, -10), -(-2 * 7 + 0 + 2 * 12))

        # Quadratic spline inside [-3, -2] (integration of linear spline with
        # one segment from (-3, -1) to (-2, 1)).
        # Extrapolates as -1 in (-inf, -3) and 2 in (-2, inf)
        spline = BSplineConstExtrapolate(
            left=-1, right=2, t=[-3, -3, -3, -2, -2, -2], c=[0, -0.5, 0, 0, 0, 0], k=2
        )
        assert np.allclose(spline.integrate(-10, 10), -1 * 7 + (-1 / 6) + 2 * 12)
        assert np.allclose(spline.integrate(10, -10), -(-1 * 7 + (-1 / 6) + 2 * 12))

    def test_derivative(self):
        # Quadratic spline inside [-3, -2] (integration of linear spline with
        # one segment from (-3, -1) to (-2, 1)).
        # Extrapolates as -1 in (-inf, -3) and 2 in (-2, inf)
        spline = BSplineConstExtrapolate(
            left=-1, right=2, t=[-3, -3, -3, -2, -2, -2], c=[0, -0.5, 0, 0, 0, 0], k=2
        )
        spline_deriv = spline.derivative()
        assert_array_equal(
            spline_deriv([-10, -3 - h, -3, -2.5, -2, -2 + h, 10]),
            np.array([0, 0, -1, 0, 0, 0, 0]),
        )
        assert_array_equal(spline_deriv([np.nan, -np.inf, np.inf]), [np.nan, 0, 0])

    def test_antiderivative(self):
        spline = BSplineConstExtrapolate(
            left=-1, right=1, t=[-3, -2, -1], c=[-0.5, 0.5], k=0
        )
        with pytest.raises(NotImplementedError):
            spline.antiderivative()
