import numpy as np
from numpy.testing import assert_array_equal
import pytest

from randomvars._utils import (
    _as_1d_finite_float,
    _sort_parallel,
    _unique_parallel,
    _assert_positive,
    _searchsorted_wrap,
    _find_nearest_ind,
    _copy_nan,
    _trapez_integral,
    _trapez_integral_cum,
    _quad_silent,
)


# `default_discrete_estimator()` is tested in `options` module


def test__as_1d_finite_float():
    with pytest.raises(ValueError, match=f"`tmp_name`.*numpy array"):
        _as_1d_finite_float({"a": None}, "tmp_name")
    with pytest.raises(ValueError, match=f"`tmp_name`.*numpy array"):
        _as_1d_finite_float({"a": None}, "tmp_name")
    with pytest.raises(ValueError, match=f"`tmp_name`.*numeric"):
        _as_1d_finite_float(["a", "a"], "tmp_name")
    with pytest.raises(ValueError, match=f"`tmp_name`.*finite values"):
        _as_1d_finite_float([0, np.nan], "tmp_name")
    with pytest.raises(ValueError, match=f"`tmp_name`.*finite values"):
        _as_1d_finite_float([0, np.inf], "tmp_name")
    with pytest.raises(ValueError, match=f"`tmp_name`.*1d array"):
        _as_1d_finite_float([[0, 1]], "tmp_name")


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


def test__assert_positive():
    with pytest.raises(ValueError, match="`tmp_name`.*negative"):
        _assert_positive(np.array([-1, 0, 1]), x_name="tmp_name")
    with pytest.raises(ValueError, match="`tmp_name`.*no positive"):
        _assert_positive(np.array([0, 0, 0]), x_name="tmp_name")


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
