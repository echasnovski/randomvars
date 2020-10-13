import warnings

import numpy as np
from scipy.integrate import quad


# %% User-facing functions
## Currently exported in `options.py`
def default_discrete_estimator(sample):
    """Default estimator of discrete distribution

    This estimator returns unique values of input as distributions values.
    Their probabilities are proportional to number of their occurrences in input.

    Parameters
    ----------
    sample : array_like
        This should be a valid input to `np.asarray()` so that its output is
        numeric.

    Returns
    -------
    x, prob : tuple with two elements
        Here `x` represents estimated values of distribution and `prob` -
        estimated probabilities.
    """
    sample = np.asarray(sample, dtype=np.float64)

    sample_is_finite = np.isfinite(sample)
    if not np.all(sample_is_finite):
        if not np.any(sample_is_finite):
            raise ValueError(
                "Input sample in discrete estimator doesn't have finite values."
            )
        else:
            warnings.warn("Input sample in discrete estimator has non-finite values.")
            sample = sample[sample_is_finite]

    vals, counts = np.unique(sample, return_counts=True)
    return vals, counts / np.sum(counts)


def default_boolean_estimator(sample):
    """Default estimator of boolean distribution

    This estimator returns proportion of `True` values after converting input
    to boolean Numpy array.

    Parameters
    ----------
    sample : array_like
        This should be a valid input to `np.asarray()` so that its output is
        boolean.

    Returns
    -------
    prob_true : number
    """
    sample = np.asarray(sample, dtype=np.bool)
    return np.mean(sample)


# %% Assertions
def _assert_equal_seq(first, second, *args, **kwargs):
    assert len(first) == len(second)
    for el1, el2 in zip(first, second):
        np.testing.assert_array_equal(el1, el2, *args, **kwargs)


# %% Array manipulations
def _as_1d_numpy(x, x_name, chkfinite=True, dtype=np.float64):
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
    dtype_chr_dict = {
        np.float64: "numeric",
        "float64": "numeric",
        np.bool: "boolean",
        "bool": "boolean",
    }
    dtype_chr = dtype_chr_dict[dtype]

    try:
        if chkfinite:
            extra_chr = " with finite values"
            res = np.asarray_chkfinite(x, dtype=dtype)
        else:
            extra_chr = ""
            res = np.asarray(x, dtype=dtype)
    except:
        raise ValueError(
            f"`{x_name}` is not convertible to {dtype_chr} numpy array{extra_chr}."
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
