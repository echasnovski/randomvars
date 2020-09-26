import warnings

import numpy as np
from scipy.integrate import quad


def _as_1d_finite_float(x, x_name):
    """Convert input to numeric numpy array and check for 1 dimension"""
    try:
        res = np.asarray_chkfinite(x, dtype=np.float64)
    except:
        raise ValueError(
            f"`{x_name}` is not convertible to numeric numpy array with finite values."
        )

    if len(res.shape) > 1:
        raise ValueError(f"`{x_name}` is not a 1d array.")

    return res


def _sort_parallel(x, y, y_name="y", warn=True):
    if len(x) != len(y):
        raise ValueError(f"Lengths of `x` and `{y_name}` do not match.")

    if not np.all(np.diff(x) >= 0):
        if warn:
            warnings.warn(
                "`x` is not sorted. "
                f"`x` and `{y_name}` are rearranged so as `x` is sorted."
            )

        x_argsort = np.argsort(x)
        x = x[x_argsort]
        y = y[x_argsort]

    return x, y


def _assert_positive(x, x_name):
    if np.any(x < 0):
        raise ValueError(f"`{x_name}` has negative values.")
    if not np.any(x > 0):
        raise ValueError(f"`{x_name}` has no positive values.")


def _searchsorted_wrap(a, v, side="left", edge_inside=True):
    """Wrapper for `np.searchsorted()` which respects `np.nan`

    Output index for every `np.nan` value in `v` is `-1`.

    Parameters
    ----------
    edge_inside: boolean, optional
        Should the most extreme edge (left for `side="left"`, right for
        `side="right"`) be treated as inside its adjacent interior interval?
        Default is `True`.
    """
    # Here call to `np.asarray()` is needed to ensure that output is not a scalar
    # (which can happen in certain cases)
    res = np.asarray(np.searchsorted(a, v, side=side))
    a = np.asarray(a)
    v = np.asarray(v)

    if edge_inside:
        if side == "right":
            # For right-most edge return coefficients from last interval
            res[v == a[-1]] = len(a) - 1
        elif side == "left":
            # For left-most edge return coefficients from first interval
            res[v == a[0]] = 1

    # Return `-1` index if element of `v` is `np.nan`
    res[np.isnan(v)] = -1

    return res


def _find_nearest_ind(x, v, side="left"):
    """Find index of nearest reference point

    For every element in `x` (arbitrary shape) find index of the nearest
    element of `v` (1d array, possibly not sorted). If there are two nearest
    reference points, choose the one from `side` ("left" or "right").

    **Note** that `nan` values are handled the same way as in `numpy.searchsorted()`.

    Parameters
    ----------
    x : array_like
        Elements for which closest ones should be found.
    v : array_like with one dimension or numerical scalar
        Array of reference points.
    side : string
        One of "left" or "right".

    Returns
    -------
    inds : array_like with shape equal to `x`'s shape
        Indices of `v`s nearest elements.
    """
    v = np.atleast_1d(v)
    if len(v.shape) != 1:
        raise ValueError("`v` should have exactly one dimension.")
    if not side in ["left", "right"]:
        raise ValueError('`side` should be one of "left" or "right".')

    if not np.all(np.diff(v) >= 0):
        v_ord = np.argsort(v)
    else:
        v_ord = np.arange(len(v))

    v_sorted = v[v_ord]
    last_v_ind = len(v) - 1

    inds = np.clip(np.searchsorted(v_sorted, x), 0, last_v_ind)
    inds_left = np.clip(inds - 1, 0, last_v_ind)

    # Modify indices to become left if it is closer than right
    ## Check for not equal left and right indices to avoid decreasing in
    ## that case, as it will lead to incorrect result
    left_isnt_right = inds_left != inds
    left_dist = x - v_sorted[inds_left]
    right_dist = v_sorted[inds] - x
    if side == "left":
        inds -= left_isnt_right & (left_dist <= right_dist)
    else:
        inds -= left_isnt_right & (left_dist < right_dist)

    return v_ord[inds]


def _trapez_integral(x, y):
    """Compute integral with trapezoidal formula."""
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def _trapez_integral_cum(x, y):
    """Compute cumulative integral with trapezoidal formula.

    Element of output represents cumulative probability **before** its left "x"
    edge.
    """
    res = np.cumsum(0.5 * np.diff(x) * (y[:-1] + y[1:]))
    return np.concatenate([[0], res])


def _quad_silent(f, a, b):
    """`quad()` with increased accuracy and direct numerical output"""
    # Ignore warnings usually resulting from reaching maximum number of
    # subdivisions without enough accuracy or bad integrand behavior
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return quad(f, a, b, limit=100)[0]
