import warnings

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline

from randomvars.options import config


# %% Array manipulations
def _as_1d_numpy(x, x_name, chkfinite=True, dtype="float64"):
    """Convert input to one-dimensional numpy array

    Parameters
    ----------
    x : array_like
    x_name : string
        Name of input to be used in possible errors.
    chkfinite : bool
        Whether to check for finite values.
    dtype : dtype
        Type of values in output array.
    """
    try:
        if chkfinite:
            extra_chr = " and finite values"
            res = np.asarray_chkfinite(x, dtype=dtype)
        else:
            extra_chr = ""
            res = np.asarray(x, dtype=dtype)
    except:
        raise TypeError(
            f"`{x_name}` is not convertible to numpy array "
            f"with {str(dtype)} dtype{extra_chr}."
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


def _unique_parallel(x, y, warn=True):
    """Remove duplicated values

    This removes duplicated values from `x` along with corresponding elements
    in second array. `x` and `y` should be the same length.
    """
    if warn:
        warnings.warn("There are duplicated values in `x`. Using first ones.")

    x, ind = np.unique(x, return_index=True)
    y = y[ind]

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


def _copy_nan(fr, to):
    return np.where(np.isnan(fr), fr, to)


# %% Math functions
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


def _is_close(x, y):
    """Check element-wise closeness

    Two numbers are considered to be equal if their absolute difference is less
    than certain tolerance value. It is computed as maximum of tolerance values
    associated with two input numbers. So to be "close", it is enough for
    numbers to differ by no more than at least one of tolerances.

    For more information about computation of tolerance associated with any
    number, see documentation of `base_tolerance` package option.

    Notes:
    - Non-finite values (`-np.inf`, `np.inf`, and `np.nan`) are according to
      IEEE rules: `np.nan` is never equal to anything, `-np.inf` and `np.inf`
      are equal only to self.
    """
    x, y = np.broadcast_arrays(x, y)
    tol_x, tol_y = _tolerance(x), _tolerance(y)
    res = np.empty_like(x, dtype="bool")

    # Finite and infinite (`-np.inf`, `np.inf`, and `np.nan`) values should be
    # processed separately
    xy_is_fin = np.isfinite(x) & np.isfinite(y)

    # Branch code for speed reasons as the most common use case is when all
    # values are finite
    if np.all(xy_is_fin):
        return np.absolute(x - y) <= np.maximum(tol_x, tol_y)
    else:
        # Process pairs with finite values
        res[xy_is_fin] = np.absolute(x[xy_is_fin] - y[xy_is_fin]) <= np.maximum(
            tol_x[xy_is_fin], tol_y[xy_is_fin]
        )

        # Process pairs with at least one infinite value
        xy_isnt_fin = ~xy_is_fin
        res[xy_isnt_fin] = x[xy_isnt_fin] == y[xy_isnt_fin]

        return res


def _is_zero(x):
    return np.absolute(x) <= config.base_tolerance


def _tolerance_coef(x):
    dtype = x.dtype if hasattr(x, "dtype") else "float64"
    one = np.array(1.0, dtype=dtype)
    return np.maximum(one, np.spacing(np.absolute(x)) / np.spacing(one))


def _tolerance(x):
    return _tolerance_coef(x) * config.base_tolerance


def _minmax(x):
    return np.nanmin(x), np.nanmax(x)


def _collapse_while_equal_fval(f, interval, side):
    """Drag one interval end towards another while having same function value

    Collapse interval (reduce width by "dragging" input side towards the other)
    in such a way that input side always has the same function value. So the
    output is a single value which is closest to the other side (maybe equal to
    it) and for which `f` has constant value on the whole segment from starting
    interval side to output.

    Collapsing is done until two interval ends are approximately equal (uses
    `base_tolerance` package option).

    Parameters
    ----------
    f : callable
    interval : List with two elements
    side : int
        Indicates side to always have the same function value: 0 for left and 1
        for right.

    Returns
    -------
    edge : numeric
    """
    val = f(interval[side])

    while not _is_close(interval[0], interval[1]):
        mid = 0.5 * (interval[0] + interval[1])
        if f(mid) == val:
            interval[side] = mid
        else:
            interval[1 - side] = mid

    # Return other side if it has reference function value. This isn't done
    # before cycle because `f` can have equal values at the input `interval`
    # but be different inside of it.
    if f(interval[1 - side]) == val:
        return interval[1 - side]
    else:
        return interval[side]


# %% Package assertions
def _assert_positive(x, x_name):
    if np.any(x < 0):
        raise ValueError(f"`{x_name}` has negative values.")
    if not np.any(x > 0):
        raise ValueError(f"`{x_name}` has no positive values.")


# %% Custom classes
class BSplineConstExtrapolate(BSpline):
    """Version of BSpline which extrapolates with constant values

    Extrapolation is right continuous (here `t` is vector of knots):
    - Value is equal to `left` in (-inf, t[0]).
    - Value is equal to `right` in [t[-1], inf).
    """

    def __init__(self, left, right, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._left = left
        self._right = right

    def __call__(self, x):
        res = super().__call__(x)
        res[x < self.t[0]] = self._left
        res[x >= self.t[-1]] = self._right

        return res

    def integrate(self, a, b):
        lim_l, lim_r = min(a, b), max(a, b)
        lim_l_in = max(lim_l, self.t[0])
        lim_r_in = min(lim_r, self.t[-1])

        res = (
            self._left * (lim_l_in - lim_l)
            + super().integrate(lim_l_in, lim_r_in)
            + self._right * (lim_r - lim_r_in)
        )
        return -res if b < a else res

    def derivative(self, nu=1):
        deriv_tck = super().derivative(nu=nu).tck
        return BSplineConstExtrapolate(0, 0, *deriv_tck)

    def antiderivative(self, nu=1):
        raise NotImplementedError(
            "Antiderivative of spline with constant extrapolation can't be simply "
            "described as spline. Implement if needed."
        )
