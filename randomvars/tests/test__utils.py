import numpy as np
from numpy.testing import assert_array_equal

from randomvars._utils import _searchsorted_wrap, _trapez_integral, _trapez_integral_cum


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
