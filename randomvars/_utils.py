import numpy as np


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
