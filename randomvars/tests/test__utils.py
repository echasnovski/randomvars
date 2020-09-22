import numpy as np
from numpy.testing import assert_array_equal
import pytest

from randomvars._utils import (
    _as_1d_finite_float,
    _sort_parallel,
    _assert_positive,
    _searchsorted_wrap,
    _trapez_integral,
    _trapez_integral_cum,
    _quad_silent,
)


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
        assert_array_equal(x_out, x[[1, 0]])
        assert_array_equal(y_out, y[[1, 0]])

    # No warning should be given if `warn=False`
    with pytest.warns(None) as record:
        x_out, y_out = _sort_parallel(x, y, warn=False)
        assert_array_equal(x_out, x[[1, 0]])
        assert_array_equal(y_out, y[[1, 0]])
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
