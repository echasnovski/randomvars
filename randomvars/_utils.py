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
