import numpy as np
from scipy.stats.distributions import rv_continuous, norm, beta, cauchy, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "../randomvars")
from rv_piecelin import rv_piecelin


#%% Functions
def coeffs_from_points(point1, point2):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    inter = point1[1] - slope * point1[0]
    return inter, slope


def line_segment_intersection(point, slope, seg_point_1, seg_point_2):
    """Return the (smallest) intersection between line and segment

    Line is defined as `y = slope*(x-point[0])+point[1]`, i.e. passing through
    point `point` and with slope `slope`. Segment is defined as connecting
    points `seg_point_1` and `seg_point_2`. **Note** that it is assumed that
    `seg_point_1[0] < seg_point_2[0]` (segment points are given in increasing
    order of x-coordinate).

    Returns
    -------
    point : Tuple representing point of intersection. `None` if no
    intersection.
    """
    # Compute parameters of line and segment's line in form `y=a*x+b`
    a1 = slope
    b1 = point[1] - slope * point[0]
    b2, a2 = coeffs_from_points(seg_point_1, seg_point_2)

    if a1 == a2:
        if b1 == b2:
            return seg_point_1[0], seg_point_1[1]
        else:
            return None
    else:
        x_inter = (b2 - b1) / (a1 - a2)
        if seg_point_1[0] <= x_inter <= seg_point_2[0]:
            return x_inter, a1 * x_inter + b1
        else:
            return None


def coneedges_segment_intersection(xy_base, slope_window, seg_point_1, seg_point_2):
    """Compute intersection of segment and 2d cone edges

    Two-dimensional cone is defined as all rays from point `xy_base` and  with
    slopes inside `slope_window` range (rays are directed to the right of
    `xy_base`).

    This function computes if there is an intersection between segment and
    both edges of this cone. In other words, it computes possible intersection
    points between segment and two lines:
    `slope_window[0]*(x-xy_base[0])+xy_base[1]` and
    `slope_window[1]*(x-xy_base[0])+xy_base[1]`.
    
    Parameters
    ----------
    xy_base : Tuple with two numbers
        Represents origin point of 2d cone.
    slope_window : Tuple with two numbers (first is less than second)
    seg_point_1 : Tuple with two numbers
        Represents start of segment.
    seg_point_2 : Tuple with two numbers
        Represents end of segment.
    
    Returns
    -------
    points : Tuple with two elements.
        Elements might be either tuple with two numbers (if there is
        intersection) or `None` (otherwise).
    """
    return tuple(
        line_segment_intersection(xy_base, slope, seg_point_1, seg_point_2)
        for slope in slope_window
    )


def tolerance_slope_window(xy_base, point, tol):
    """ Compute slope window for rays to be within tolerance of supplied point

    Computes slope window of 2d cone with base point `xy_base` and which passes
    through points `(point[0], point[1]-tol)` and `(point[0], point[1]+tol)`.
    """
    return tuple(
        (point[1] - xy_base[1] + extra) / (point[0] - xy_base[0])
        for extra in [-tol, tol]
    )


def intersect_inervals(interval1, interval2):
    return max(interval1[0], interval2[0]), min(interval1[1], interval2[1])


def regrid_maxtol(x, y, tol=1e-3):
    """Regrid with maximum tolerance

    Regrid input xy-greed so that maximum difference between points on output
    piecewise-linear function and xy-greed is less than `tol` (currently it is
    always equal to `tol`).
    
    Parameters
    ----------
    x : Numpy numeric array.
    y : Numpy numeric array.
    tol : Single number, optional
        Tolerance, by default 1e-3
    
    Returns
    -------
    xy_grid : Tuple with two numpy numeric arrays with same lengths
    """
    if len(x) <= 2:
        return x, y

    x_res, y_res = [np.empty_like(i, dtype=np.float64) for i in [x, y]]

    # First point should also be in result
    x_res[0], y_res[0] = x[0], y[0]
    res_i = 1

    # Initialize base point and slope window
    xy_base = (x[0], y[0])
    slope_window = tolerance_slope_window(xy_base, (x[1], y[1]), tol)

    cur_i = 2
    while cur_i < len(x):
        seg_start = (x[cur_i - 1], y[cur_i - 1])
        seg_end = (x[cur_i], y[cur_i])

        # Compute candidate result points by looking if edges of current base cone
        # intersect with current segment
        xy_cand = coneedges_segment_intersection(
            xy_base, slope_window, seg_start, seg_end
        )
        xy_not_none = [xy for xy in xy_cand if xy is not None]

        if len(xy_not_none) > 0:
            # If there is several intersections, take the left one (which
            # probably shouldn't be possible; added just in case)
            x_new, y_new = min(xy_not_none, key=lambda xy: xy[0])

            # Write new point
            x_res[res_i], y_res[res_i] = x_new, y_new
            res_i += 1

            # Update base point
            xy_base = (x_new, y_new)

            # Update slope window
            ## If new point is exactly at the end of the current segment,
            ## move one segment further and use its end to compute slope window
            if xy_base[0] == seg_end[0]:
                cur_i += 1
                if cur_i >= len(x):
                    break

            # End of the "current" segment (either this iteration's current
            # segment or next segment in case `xy_base` is moved to current
            # segment end) now defines slope window
            slope_window = tolerance_slope_window(xy_base, (x[cur_i], y[cur_i]), tol)
        else:
            # Update slope window by using intersection of current slope window
            # and slope window of segment end. Intersection is used because in
            # order to maintain maximum error within tolerance rays should pass
            # inside both current 2d cone and 2d cone defined by end of current
            # segment
            seg_end_slope_window = tolerance_slope_window(xy_base, seg_end, tol)
            slope_window = intersect_inervals(slope_window, seg_end_slope_window)

        cur_i += 1

    # If the last point wasn't added (most of the times), add it
    if x_res[res_i - 1] != x_res[-1]:
        x_res[res_i], y_res[res_i] = x[-1], y[-1]
        res_i += 1

    if res_i == len(x):
        # If output has the same number of points as input, return input (as it
        # is always at least not worse)
        return x, y
    else:
        return x_res[:res_i], y_res[:res_i]


def trapez_integral(x, y):
    """ Compute integral with trapezoidal formula.
    """
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def cons_cos(x, y):
    """Compute cosine of consecutive vectors

    Xy-grid, created with `x` and `y`, defines a sequence of vectors:
    `(x[i]-x[i-1]; y[i]-y[i-1])`. This function computes cosine of angles
    between consecutive pairs between them.  It is assumed that `x` is sorted
    increasingly, but no checks is done.

    Inputs `x` and `y` should be the same length with at least 3 elements.

    Parameters
    ----------
    x : numpy numeric array
        It is assumed that it is sorted increasingly.
    y : numpy numeric array

    Returns
    -------
    cos : numpy numeric array with length `len(x)-2`
        Cosine of right-aligned vectors, defined by xy-grid

    Examples
    --------
    >>> cons_cos(np.arange(6), np.array([0, 1, 1, 0, 1, 2]))
    array([0.70710678, 0.70710678, 0.        , 1.        ])
    """

    def dot_prod(vec_1, vec_2):
        return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]

    vec_x = np.diff(x)
    vec_y = np.diff(y)

    vec_left = (vec_x[:-1], vec_y[:-1])
    vec_right = (vec_x[1:], vec_y[1:])

    res = dot_prod(vec_left, vec_right) / np.sqrt(
        dot_prod(vec_left, vec_left) * dot_prod(vec_right, vec_right)
    )

    return res


def sharpness(x, y):
    """Compute sharpness of piecewise-linear vertices

    Here "sharpness" is a number between 0 and 1, where 0 represents that point
    lies on a straight line and 1 - that it is an edge of extreme peak.

    It is assumed that `x` and `y` have the same length. First and last elements
    by default have sharpness of 0.

    Parameters
    ----------
    x : numpy numeric array
        It is assumed that it is sorted increasingly.
    y : numpy numeric array

    Returns
    -------
    sharpness: numpy array of the same length as `x` and `y`.

    Examples
    --------
    >>> sharpness(np.arange(6), np.array([0, 1, 1, 0, 1, 2]))
    array([0.        , 0.14644661, 0.14644661, 0.5       , 0.        ,
           0.        ])
    """
    # Cosine between consecutive vectors has value `1` if vectors are colinear
    # (i.e. if their shared point lies on a straight line) and `-1` if they
    # aligned but have opposite directions (i.e. shared point is an extreme
    # peak)
    vec_cos = cons_cos(x, y)

    return np.concatenate([[0], 0.5 * (1 - vec_cos), [0]])


def abs_curvature(x, y):
    deriv = np.gradient(y, x)
    deriv2 = np.gradient(deriv, x)
    return np.abs(deriv2) / (1 + deriv ** 2) ** 1.5


def piecelin_quantiles(x, y, probs):
    rv = rv_piecelin(x, y)
    return rv.ppf(probs)


def regress_to_mean(x, y, alpha=0.5):
    integral_mean = trapez_integral(x, y) / (x[-1] - x[0])
    return (1 - alpha) * y + alpha * integral_mean


def regrid_curvature(x, y, n_grid, alpha=0.5):
    curv = abs_curvature(x, y)
    curv_regr = regress_to_mean(x, curv, alpha)
    x_new = piecelin_quantiles(x, curv_regr, np.linspace(0, 1, n_grid))
    y_new = np.interp(x_new, x, y)
    return x_new, y_new


def regrid_equidist(x, y, n_grid):
    x_new = np.linspace(x[0], x[-1], n_grid)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new


def dist_grid(grid_base, grid_new, method=None):
    if method is None:
        method = "meanabs"

    diff = np.interp(grid_base[0], grid_new[0], grid_new[1]) - grid_base[1]

    if method == "maxabs":
        return np.max(np.abs(diff))
    elif method == "meanabs":
        return np.mean(np.abs(diff))
    elif method == "meanpmaxabs":
        abs_diff = np.abs(diff)
        return np.mean(abs_diff) + np.max(abs_diff)
    else:
        raise ValueError


def regrid_optimize(x, y, n_grid, maxiter=np.inf, fit_method=None):
    if len(x) == 2:
        return x

    supp = tuple(x[[0, -1]])

    def fit_func(x_inn):
        x_fit = np.concatenate(([supp[0]], x_inn, [supp[1]]))
        y_fit = np.interp(x_fit, x, y)
        return dist_grid((x, y), (x_fit, y_fit), method=fit_method)

    x_init = np.linspace(x[0], x[-1], n_grid)[1:-1]
    bounds = [supp] * (n_grid - 2)
    x_inn_res = minimize(
        fit_func, x_init, bounds=bounds, options={"maxiter": maxiter}
    ).x
    x_res = np.concatenate(([supp[0]], x_inn_res, [supp[1]]))
    y_res = np.interp(x_res, x, y)

    return x_res, y_res


#%% Experiments
# Base grid
# ## Base grid from distribution
# # dist = norm
# # dist = beta(10, 20)
# # dist = cauchy
# dist = chi2(2)
#
# n_grid = 10001
# supp = dist.ppf([1e-6, 1 - 1e-6])
# x = np.linspace(supp[0], supp[1], n_grid)
# y = dist.pdf(x)

## Base grid from manual data
x_base = np.array([0, 2, 4, 5, 6, 7])
y_base = np.array([0, 1, 0, 0, 1, 0])
y_base = y_base / trapez_integral(x_base, y_base)

n_grid = 10001
supp = x_base[[0, -1]]
x = np.linspace(supp[0], supp[1], n_grid)
y = np.interp(x, x_base, y_base)

# Regridding
x_maxtol, y_maxtol = regrid_maxtol(x, y, tol=1e-3)
n_grid_new = len(x_maxtol)
x_equi, y_equi = regrid_equidist(x, y, n_grid_new)
x_curv, y_curv = regrid_curvature(x, y, n_grid_new)
x_optim, y_optim = regrid_optimize(x, y, n_grid_new)

dist_grid((x, y), (x_maxtol, y_maxtol), method="maxabs")
dist_grid((x, y), (x_equi, y_equi), method="maxabs")
dist_grid((x, y), (x_curv, y_curv), method="maxabs")
dist_grid((x, y), (x_optim, y_optim), method="maxabs")

dist_grid((x, y), (x_maxtol, y_maxtol), method="meanabs")
dist_grid((x, y), (x_equi, y_equi), method="meanabs")
dist_grid((x, y), (x_curv, y_curv), method="meanabs")
dist_grid((x, y), (x_optim, y_optim), method="meanabs")

trapez_integral(x_maxtol, y_maxtol)
trapez_integral(x_equi, y_equi)
trapez_integral(x_curv, y_curv)
trapez_integral(x_optim, y_optim)

%timeit regrid_maxtol(x, y, tol=1e-3)
%timeit regrid_equidist(x, y, n_grid_new)
%timeit regrid_curvature(x, y, n_grid_new)
%timeit regrid_optimize(x, y, n_grid_new)

# Plot regridding outputs
fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x_equi, y_equi, color="red", marker="o")
ax.plot(x_curv, y_curv, color="green", marker="o")
ax.plot(x_optim, y_optim, color="blue", marker="o")
ax.plot(x_maxtol, y_maxtol, color="magenta", marker="o")

# Plot absolute and regressed-to-mean curvatures
abs_curv = abs_curvature(x, y)
abs_curv_regr = regress_to_mean(x, abs_curv, alpha=0.5)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, abs_curv, color="blue")
ax.plot(x, abs_curv_regr, color="green")

# Study effect of "regress-to-mean" parameter on approximation quality
alpha_grid = np.linspace(0, 1, 11)
[
    dist_grid((x, y), regrid_curvature(x, y, n_grid_new, alpha=a), method="maxabs")
    for a in alpha_grid
]
