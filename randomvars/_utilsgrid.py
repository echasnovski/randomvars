import numpy as np

import randomvars._utils as utils
import randomvars.options as op

# %% Conversion
# There were different other approaches to Cont-Disc conversion, which were
# decided to be less appropriate:
# - In Cont-Disc construct discrete distribution with the same x-grid to be the
#   closest to input continuous CDF in terms of some metric ("L1" or "L2").
#   These were discarded because they were not invertible and hence not really
#   possible to create appropriate Disc-Cont conversion. The problem was that
#   during inverse conversion there were negative values in y-grid, which is an
#   additional problem. For example, `x = [0, 1]`, `p = [0.9, 0.1]`.
# - Another idea of Cont-Disc conversion was along the following lines:
#     - Assume there are many elements sampled from input distribution.
#     - For every sample element find the closest one among input x-grid.
#     - Take sample probability of x-grid elements as ratio of number of times
#       it was the closest and number of all points.
#     - Probability of element in x-grid is a limit of sample probabilities. Those
#       can be computed directly by computing probability of Voronoi intervals
#       (with ends at midpoints of adjacent intervals).
#   This turned out to be a previous approach with "L1" metric, which is not
#   invertible.


def _y_from_xp(x, p):
    """Compute y-grid from xp-grid

    Compute y-grid which together with input x-grid is dual to input xp-grid.
    Duality is defined in terms of maximum likelihood estimation. Output
    xy-grid maximizes weighted log-likelihood `sum(p * log(y))` subject to
    integration constraint on xy-grid (`0.5 * sum((x[1:] - x[:-1]) * (y[1:] +
    y[:-1])) = 1`).

    Notes:
    - Points with zero p-elements affect the output y-grid: they indicate that
      in that region probability should be low (corresponding elements of
      y-grid will be zero). This is somewhat counterintuitive, as presence of
      zero probabilities doesn't change input discrete variable, but affects
      output continuous one.
    """
    return p / _convert_coeffs(x)


def _p_from_xy(x, y):
    """Compute p-grid from xy-grid

    Compute p-grid which together with input x-grid is dual to input xy-grid.
    Duality is defined in terms of maximum likelihood estimation of xy-grid.
    Output xp-grid is the one, for which input xy-grid maximizes weighted
    log-likelihood `sum(p * log(y))` subject to integration constraint on
    xy-grid (`0.5 * sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) = 1`). This
    approach is taken to be inverse of y-from-p conversion.

    Notes:
    - Points with zero y-elements result into zero p-elements.
    """
    return y * _convert_coeffs(x)


def _convert_coeffs(x):
    """These are coefficients of y-grid when computing integral using
    trapezoidal rule"""
    x_ext = np.concatenate(([x[0]], x, [x[-1]]))
    return 0.5 * (x_ext[2:] - x_ext[:-2])


# %% Stacking
def _stack_xy(xy_seq):
    """Stack xy-grids

    Here "stack xy-grids" means "compute xy-grid which represents sum of all
    input xy-grids".

    Compute xy-grid representing piecewise-linear function which is a sum of
    input xy-grids. As every xy-grid (possibly) has discontinuities at edges
    (if y-value is not zero) output uses "grounded" versions of xy-grids: the
    ones which represent explicit piecewise-linear on the whole real line.
    Grounding is done only where it is necessary: if certain edge will be an
    edge of output xy-grid, it is not grounded (this also helps sustaining the
    exact support of output).

    Output x-grid consists of all unique values from all input x-grids. Output
    y-grid is computed as sum of interpolations at output x-grid for all input
    y-grids.

    Parameters
    ----------
    xy_seq : sequence
        Sequence of xy-grids.
    """
    # Determine grounding direction for every xy-grid so that resulting edges
    # of output don't get unnecessary grounding
    ground_dir = _compute_stack_ground_dir(xy_seq)

    # Grounding is needed to ensure that `x_tbl` doesn't affect its outside
    xy_grounded_seq = [_ground_xy(xy, dir) for xy, dir in zip(xy_seq, ground_dir)]

    # Compute stacked x-grid as unique values of all grounded x-grids
    ## This relies on fact that `np.unique()` returns sorted output
    x = np.unique(np.concatenate([xy[0] for xy in xy_grounded_seq]))

    # Stack xy-grids by evaluating grounded versions at output x-grid
    y_list = [np.interp(x, x_in, y_in) for x_in, y_in in xy_grounded_seq]
    y = np.array(y_list).sum(axis=0)

    return x, y


def _compute_stack_ground_dir(xy_seq):
    output_range = utils._minmax(np.concatenate([xy[0] for xy in xy_seq]))
    ground_left = [~utils._is_close(x[0], output_range[0]) for x, _ in xy_seq]
    ground_right = [~utils._is_close(x[-1], output_range[-1]) for x, _ in xy_seq]

    ground_dict = {
        (False, False): "none",
        (True, False): "left",
        (False, True): "right",
        (True, True): "both",
    }

    return [ground_dict[gr] for gr in zip(ground_left, ground_right)]


def _ground_xy(xy, direction=None):
    """Update xy-grid to represent explicit piecewise-linear function

    Implicitly xy-grid represents piecewise-linear function in the following way:
    - For points inside `[x[0]; x[-1]]` (support) output is a linear
      interpolation.
    - For points outside support output is zero.

    This function transforms xy-grid so that output can be computed as a direct
    linear interpolation. This is done by possibly approximating "jumps" at the
    edge(s) of support. Approximation is performed by introducing a linear
    smoothing of a jump: one point close to edge is added outside of support
    and, in case there isn't a "close" one present, one on the inside.
    Closeness is determined via "small_width" option. Y-values are: zero for
    outside, respective density value for inside. Then y-value of edge knot is
    modified so as to preserve total probability.

    Notes:
    - If edge is already zero, then no grounding is done.

    Parameters
    ----------
    xy : tuple with two elements
    direction : string or None
        Can be one of `"both"`, `"left"`, `"right"`, `"none"` or `None`.
        Controls which edge(s) should be grounded (if any).
    """
    if (direction is None) or (direction == "none"):
        return xy

    x, y = xy
    w = op.get_option("small_width")

    ground_left = (direction in ["left", "both"]) and (not utils._is_zero(y[0]))
    ground_right = (direction in ["right", "both"]) and (not utils._is_zero(y[-1]))

    xy_fun = lambda t: np.interp(t, x, y, left=0.0, right=0.0)
    x_res, y_res = x, y

    if ground_left:
        x_diff = x[1] - x[0]

        # Using `2*w` instead of `w` to avoid numerical representation issues
        if x_diff > 2 * w:
            # Case when inner point should be added because there is no "close"
            # knot in input data
            x_res = np.concatenate(([x[0] - w, x[0], x[0] + w], x_res[1:]))
            y_res = np.concatenate(([0.0, 0.5 * y[0], xy_fun(x[0] + w)], y_res[1:]))
        else:
            # Case when inner point shouldn't be added
            x_res = np.concatenate(([x[0] - w, x[0]], x_res[1:]))
            y_res = np.concatenate(([0.0, y[0] * x_diff / (x_diff + w)], y_res[1:]))

    if ground_right:
        x_diff = x[-1] - x[-2]

        # Using `2*w` instead of `w` to avoid numerical representation issues
        if x_diff > 2 * w:
            # Case when inner point should be added because there is no "close"
            # knot in input data
            x_res = np.concatenate((x_res[:-1], [x[-1] - w, x[-1], x[-1] + w]))
            y_res = np.concatenate((y_res[:-1], [xy_fun(x[-1] - w), 0.5 * y[-1], 0.0]))
        else:
            # Case when inner point shouldn't be added
            x_res = np.concatenate((x_res[:-1], [x[-1], x[-1] + w]))
            y_res = np.concatenate((y_res[:-1], [y[-1] * x_diff / (x_diff + w), 0.0]))

    return x_res, y_res
