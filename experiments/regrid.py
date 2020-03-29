import numpy as np
from scipy.stats.distributions import rv_continuous, norm, beta, cauchy, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "../randomvars")
from rv_piecelin import rv_piecelin


#%% Functions
# All functions related to `regrid_maxtol()` are written using tuples for
# function arguments as much as possible to increase execution speed.
# This showed significant increase for most tupical cases (~10%).
def coneedges_segment_intersection(
    base_x, base_y, slope_min, slope_max, seg1_x, seg1_y, seg2_x, seg2_y
):
    """Compute intersection of segment and 2d cone edges

    Two-dimensional cone is defined as all rays from point `(base_x, base_y)`
    and  with slopes inside `[slope_min, slope_max]` range (rays are directed
    to the right of origin). Segment connects point `(seg1_x, seg1_y)` and
    `(seg2_x, seg2_y)`.

    This function computes if there is an intersection between segment and
    some edge of the cone. In other words, it computes possible intersection
    points between segment and two lines: `slope_min*(x-base_x)+base_y` and
    `slope_max*(x-base_x)+base_y`.

    **Note** that for correctness of algorithm, it is assumed that `(seg1_x,
    seg1_y)` is strictly inside the cone (not on edge). In other words, slope
    of line through `(base_x, base_y)` and `(seg1_x, seg1_y)` lies strictly
    inside `(slope_min, slope_max)` interval.  This enables using simplified
    algorithm for computing if there is an intersection (which terminates early
    if there is no intersection, which in turn increases execution speed) and
    its coordinates. For example, it implies that there can be only one
    intersection between segment and cone edges.

    Parameters
    ----------
    base_x, base_y : Numbers for x and y coordinates of 2d cone origin point
    slope_min, slope_max : Numbers for minimum and maximum values of slope
    (edges of 2d cone)
    seg1_x, seg1_y : Numbers for x and y coordinates of segment start
    seg2_x, seg2_y : Numbers for x and y coordinates of segment end

    Returns
    -------
    point : Tuple with two elements (if there is intersection) or `None`
    (otherwise).
    """
    seg_slope_1 = (seg1_y - base_y) / (seg1_x - base_x)
    seg_slope_2 = (seg2_y - base_y) / (seg2_x - base_x)

    # This part uses assumption that `seg_slope_1` lies inside `(slope_min,
    # slope_max)`. It enables easy checking if cone edge intersects segment by
    # comparing edge's slope with segment's slopes.
    if (seg_slope_1 <= slope_min) or (seg_slope_2 <= slope_min):
        a1 = slope_min
    elif (seg_slope_1 >= slope_max) or (seg_slope_2 >= slope_max):
        a1 = slope_max
    else:
        return None

    # This part executes only if there is an intersection between some cone
    # edge (with base slope `a1`) and segment. Thus there is no need to check
    # if intersection of lines lie inside segment.
    b1 = base_y - a1 * base_x
    a2 = (seg2_y - seg1_y) / (seg2_x - seg1_x)
    b2 = seg1_y - a2 * seg1_x

    x_res = (b2 - b1) / (a1 - a2)
    y_res = a2 * x_res + b2
    return x_res, y_res


def tolerance_slope_window(base_x, base_y, point_x, point_y, tol):
    """ Compute slope window for rays to be within tolerance of supplied point

    Computes slope window of 2d cone with base point `(base_x, base_y)` and
    which passes through points `(point_x, point_y-tol)` and `(point_x,
    point_y+tol)`.
    """
    slope_min = (point_y - base_y - tol) / (point_x - base_x)
    slope_max = (point_y - base_y + tol) / (point_x - base_x)

    return slope_min, slope_max


def intersect_intervals(inter1_min, inter1_max, inter2_min, inter2_max):
    """Compute intersection of intervals

    Computes intersections of intervals `(inter1_min, inter1_max)` and
    `(inter2_min, inter2_max)`. Basically, the output is `(max(inter1_min,
    inter2_min), min(inter1_max, inter2_max))` but optimized for better
    execution speed.
    """
    if inter1_min <= inter2_min:
        res_min = inter2_min
    else:
        res_min = inter1_min

    if inter1_max <= inter2_max:
        res_max = inter1_max
    else:
        res_max = inter2_max

    return res_min, res_max


def regrid_maxtol(x, y, tol=1e-3):
    """Regrid with maximum tolerance

    Regrid input xy-greed so that maximum difference between points on output
    piecewise-linear function and input xy-greed is not more than `tol`
    (currently it is always equal to `tol`).

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

    x_res = [x[0]]
    y_res = [y[0]]

    # Initialize base point and slope window
    base_x = x[0]
    base_y = y[0]
    slope_min, slope_max = tolerance_slope_window(base_x, base_y, x[1], y[1], tol)

    cur_i = 2
    while cur_i < len(x):
        seg_end_x = x[cur_i]
        seg_end_y = y[cur_i]

        # Compute candidate result point by looking if any edge of current base
        # cone intersect current segment
        xy_cand = coneedges_segment_intersection(
            base_x = base_x,
            base_y = base_y,
            slope_min = slope_min,
            slope_max = slope_max,
            seg1_x = x[cur_i - 1],
            seg1_y = y[cur_i - 1],
            seg2_x = seg_end_x,
            seg2_y = seg_end_y,
        )

        if xy_cand is None:
            # Update slope window by using intersection of current slope window
            # and slope window of segment end. Intersection is used because in
            # order to maintain maximum error within tolerance rays should pass
            # inside both current 2d cone and 2d cone defined by end of current
            # segment
            seg_end_slope_min, seg_end_slope_max = tolerance_slope_window(
                base_x, base_y, seg_end_x, seg_end_y, tol
            )
            slope_min, slope_max = intersect_intervals(
                slope_min, slope_max, seg_end_slope_min, seg_end_slope_max
            )
        else:
            # Write new point
            x_res.append(xy_cand[0])
            y_res.append(xy_cand[1])

            # Update base point
            base_x, base_y = xy_cand

            # Update slope window
            ## If new point is exactly at the end of the current segment,
            ## (detected with `>=` instead of `==` to account for possible
            ## numerical representation error) move one segment further and
            ## use its end to compute slope window
            if base_x >= seg_end_x:
                cur_i += 1
                if cur_i >= len(x):
                    break

            # End of the "current" segment (either this iteration's current
            # segment or next segment in case `xy_base` is moved to current
            # segment end) now defines slope window
            slope_min, slope_max = tolerance_slope_window(
                base_x, base_y, x[cur_i], y[cur_i], tol
            )

        cur_i += 1

    # If the last point wasn't added (as is most of the times), add it
    if x_res[-1] != x[-1]:
        x_res.append(x[-1])
        y_res.append(y[-1])

    if len(x_res) == len(x):
        # If output has the same number of points as input, return input (as it
        # is always a better choice)
        return x, y
    else:
        return np.array(x_res), np.array(y_res)


def trapez_integral(x, y):
    """ Compute integral with trapezoidal formula.
    """
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


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


def dist_grid_cdf(grid_base, grid_new, method=None):
    base_rv = rv_piecelin(*grid_base)
    new_rv = rv_piecelin(*grid_new)

    return dist_grid(
        (grid_base[0], base_rv.cdf(grid_base[0])),
        (grid_new[0], new_rv.cdf(grid_new[0])),
        method=method
    )


def dist_grid_fun(fun, grid, method=None, n_inner_points=10):
    x, y = grid
    test_arr = [
        np.linspace(x[i], x[i+1], n_inner_points+2) for i in np.arange(len(x)-1)
    ]
    x_test = np.unique(np.concatenate(test_arr))
    y_test = fun(x_test)

    return dist_grid((x_test, y_test), grid, method)


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
## Base grid from distribution
dist = norm
# dist = beta(10, 20)
# dist = cauchy
# dist = chi2(2)
# dist = chi2(1)

n_grid = 10001
supp = dist.ppf([1e-6, 1 - 1e-6])
x = np.linspace(supp[0], supp[1], n_grid)
# y = dist.pdf(x)
y = np.gradient(dist.cdf(x), x)

# ## Base grid from manual data
# x_base = np.array([0, 2, 4, 5, 6, 7])
# y_base = np.array([0, 1, 0, 0, 1, 0])
# y_base = y_base / trapez_integral(x_base, y_base)

# n_grid = 10001
# supp = x_base[[0, -1]]
# x = np.linspace(supp[0], supp[1], n_grid)
# y = np.interp(x, x_base, y_base)

# Regridding
integr_tol = 1e-3
tol = integr_tol / (x[-1] - x[0])
print(f"tol={tol}")
x_maxtol, y_maxtol = regrid_maxtol(x, y, tol=tol)
n_grid_new = len(x_maxtol)
print(f"n_grid_new={n_grid_new}")
x_equi, y_equi = regrid_equidist(x, y, n_grid_new)
x_curv, y_curv = regrid_curvature(x, y, n_grid_new)
x_optim, y_optim = regrid_optimize(x, y, n_grid_new)

regriddings = {
    "maxtol": (x_maxtol, y_maxtol),
    "equi": (x_equi, y_equi),
    "curv": (x_curv, y_curv),
    "optim": (x_optim, y_optim),
}

for meth, grid in regriddings.items():
    grid_dist = dist_grid((x, y), grid, method="maxabs")
    print(f"'maxabs' distance for '{meth}' method: {grid_dist}")

for meth, grid in regriddings.items():
    grid_dist = dist_grid((x, y), grid, method="meanabs")
    print(f"'meanabs' distance for '{meth}' method: {grid_dist}")

for meth, grid in regriddings.items():
    grid_dist = dist_grid_cdf((x, y), grid, method="maxabs")
    print(f"'maxabs cdf' distance for '{meth}' method: {grid_dist}")

for meth, grid in regriddings.items():
    grid_dist = dist_grid_fun(dist.pdf, grid, method="maxabs")
    print(f"'maxabs fun' distance for '{meth}' method: {grid_dist}")

for meth, grid in regriddings.items():
    integral_extra = trapez_integral(*grid) - 1
    print(f"Trapez. integral extra for '{meth}' method: {integral_extra}")

%timeit regrid_maxtol(x, y, tol=tol)
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
