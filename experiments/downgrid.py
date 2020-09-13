import numpy as np
from scipy.stats.distributions import rv_continuous, norm, beta, cauchy, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

# This uses "development" version of `Cont`. instead of "installed in
# current virtual environment" version.
sys.path.insert(0, "../randomvars")
from _cont import Cont

from downgrid_maxtol import downgrid_maxtol


#%% Functions
# All functions related to `downgrid_maxtol()` are written without using tuples
# for function arguments as much as possible to increase execution speed. This
# showed significant increase for most tupical cases (~10%).
def is_segment_inside_cone(
    base_x, base_y, slope_min, slope_max, seg1_x, seg1_y, seg2_x, seg2_y
):
    """Compute if segment lies inside 2d closed cone

    Two-dimensional closed cone is defined as all rays from point `(base_x,
    base_y)` and  with slopes inside `[slope_min, slope_max]` range (rays are
    directed to the right of origin). Segment connects point `(seg1_x, seg1_y)`
    and `(seg2_x, seg2_y)`.

    This function computes if whole segment lies inside cone (even if it
    touches some edge).

    Parameters
    ----------
    base_x, base_y : Numbers for x and y coordinates of 2d cone origin point
    slope_min, slope_max : Numbers for minimum and maximum values of slope
    (edges of 2d cone)
    seg1_x, seg1_y : Numbers for x and y coordinates of segment start
    seg2_x, seg2_y : Numbers for x and y coordinates of segment end

    Returns
    -------
    is_inside : Boolean value indicating if whole segment lies inside cone.
    """
    seg_slope_1 = (seg1_y - base_y) / (seg1_x - base_x)
    seg_slope_2 = (seg2_y - base_y) / (seg2_x - base_x)

    # Segment lies inside cone if its both ends' slopes (computed with respect to
    # cone's base point) lie inside `[slope_min, slope_max]`
    if (
        (seg_slope_1 >= slope_min)
        and (seg_slope_1 <= slope_max)
        and (seg_slope_2 >= slope_min)
        and (seg_slope_2 <= slope_max)
    ):
        return True
    else:
        return False


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


def downgrid_maxtol_python(x, y, tol=1e-3):
    """Downgrid with maximum tolerance

    Downgrid input xy-grid so that maximum difference between points on output
    piecewise-linear function and input xy-grid is not more than `tol`. Output
    xy-grid is a subset of input xy-grid. **Note** that first and last point is
    always inside output xy-grid.

    Parameters
    ----------
    x : Numpy numeric array.
    y : Numpy numeric array.
    tol : Single number, optional
        Tolerance, by default 1e-3

    Returns
    -------
    xy_grid : Tuple with two numpy numeric arrays with same lengths
        Subset of input xy-grid which differs from it by no more than `tol`.
    """
    if (len(x) <= 2) or (tol == 0):
        return x, y

    # First point is always inside output grid
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

        # Compute if segment lies inside current base cone. If it does, then it
        # can be skipped. It it goes out of the current base cone, it means
        # that skipping will introduce error strictly more than `tol`, so
        # adding current segment start to output xy-grid is necessary.
        segment_is_inside = is_segment_inside_cone(
            base_x=base_x,
            base_y=base_y,
            slope_min=slope_min,
            slope_max=slope_max,
            seg1_x=x[cur_i - 1],
            seg1_y=y[cur_i - 1],
            seg2_x=seg_end_x,
            seg2_y=seg_end_y,
        )

        if segment_is_inside:
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
            # Write segment start as new point in output
            seg_start_x = x[cur_i - 1]
            seg_start_y = y[cur_i - 1]
            x_res.append(seg_start_x)
            y_res.append(seg_start_y)

            # Update base point to be current segment start
            base_x, base_y = seg_start_x, seg_start_y

            # Update slope window to be slope window of current segment end
            slope_min, slope_max = tolerance_slope_window(
                base_x, base_y, seg_end_x, seg_end_y, tol
            )

        cur_i += 1

    # Last point is always inside output grid
    x_res.append(x[-1])
    y_res.append(y[-1])

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
    rv = Cont(x, y)
    return rv.ppf(probs)


def regress_to_mean(x, y, alpha=0.5):
    integral_mean = trapez_integral(x, y) / (x[-1] - x[0])
    return (1 - alpha) * y + alpha * integral_mean


def downgrid_curvature(x, y, n_grid, alpha=0.5):
    curv = abs_curvature(x, y)
    curv_regr = regress_to_mean(x, curv, alpha)
    x_new = piecelin_quantiles(x, curv_regr, np.linspace(0, 1, n_grid))
    y_new = np.interp(x_new, x, y)
    return x_new, y_new


def downgrid_equidist(x, y, n_grid):
    x_new = np.linspace(x[0], x[-1], n_grid)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new


def dist_grid(grid_base, grid_new, method=None):
    diff = np.interp(grid_base[0], grid_new[0], grid_new[1]) - grid_base[1]
    return diff_summary(diff)


def dist_grid_cdf(grid_base, grid_new, method=None):
    base_rv = Cont(*grid_base)
    new_rv = Cont(*grid_new)

    diff = new_rv.cdf(grid_base[0]) - base_rv.cdf(grid_base[0])
    return diff_summary(diff)


def diff_summary(diff, method=None):
    if method is None:
        method = "maxabs"

    if method == "maxabs":
        return np.max(np.abs(diff))
    elif method == "meanabs":
        return np.mean(np.abs(diff))
    elif method == "meanpmaxabs":
        abs_diff = np.abs(diff)
        return np.mean(abs_diff) + np.max(abs_diff)
    else:
        raise ValueError


def augment_x_grid(x, n_inner_points=10):
    test_arr = [
        np.linspace(x[i], x[i + 1], n_inner_points + 2) for i in np.arange(len(x) - 1)
    ]
    return np.unique(np.concatenate(test_arr))


def dist_pdf_fun(pdf, grid, method=None, n_inner_points=10):
    x_test = augment_x_grid(grid[0])
    y_test = pdf(x_test)

    return dist_grid((x_test, y_test), grid, method)


def dist_cdf_fun(cdf, grid, method=None, n_inner_points=10):
    x_test = augment_x_grid(grid[0])
    rv_grid = Cont(*grid)

    diff = cdf(x_test) - rv_grid.cdf(x_test)

    return diff_summary(diff, method)


def downgrid_optimize(x, y, n_grid, maxiter=np.inf, fit_method=None):
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
# dist = norm
# dist = beta(10, 20)
# dist = cauchy
# dist = chi2(2)
dist = chi2(1)

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

# Downgridding
integr_tol = 1e-3
tol = integr_tol / (x[-1] - x[0])
print(f"tol={tol}")
x_maxtol, y_maxtol = downgrid_maxtol(x, y, tol=tol)
x_maxtol_python, y_maxtol_python = downgrid_maxtol_python(x, y, tol=tol)
n_grid_new = len(x_maxtol)
print(f"n_grid_new={n_grid_new}")
x_equi, y_equi = downgrid_equidist(x, y, n_grid_new)
x_curv, y_curv = downgrid_curvature(x, y, n_grid_new)
x_optim, y_optim = downgrid_optimize(x, y, n_grid_new)

np.allclose(x_maxtol, x_maxtol_python)
np.allclose(y_maxtol, y_maxtol_python)

downgriddings = {
    "maxtol": (x_maxtol, y_maxtol),
    "equi": (x_equi, y_equi),
    "curv": (x_curv, y_curv),
    "optim": (x_optim, y_optim),
}

# Grid metrics
for meth, grid in downgriddings.items():
    grid_dist = dist_grid((x, y), grid, method="maxabs")
    print(f"'maxabs' distance for '{meth}' method: {grid_dist}")

for meth, grid in downgriddings.items():
    grid_dist = dist_grid((x, y), grid, method="meanabs")
    print(f"'meanabs' distance for '{meth}' method: {grid_dist}")

for meth, grid in downgriddings.items():
    grid_dist = dist_grid_cdf((x, y), grid, method="maxabs")
    print(f"'maxabs cdf' distance for '{meth}' method: {grid_dist}")

# Distribution functions' metrics
for meth, grid in downgriddings.items():
    grid_dist = dist_pdf_fun(dist.pdf, grid, method="maxabs")
    print(f"'maxabs pdf_fun' distance for '{meth}' method: {grid_dist}")

for meth, grid in downgriddings.items():
    grid_dist = dist_cdf_fun(dist.cdf, grid, method="maxabs")
    print(f"'maxabs cdf_fun' distance for '{meth}' method: {grid_dist}")

# Integral metric
for meth, grid in downgriddings.items():
    integral_extra = trapez_integral(*grid) - 1
    print(f"Trapez. integral extra for '{meth}' method: {integral_extra}")

# Execution timings
%timeit downgrid_maxtol(x, y, tol=tol)
%timeit downgrid_maxtol_python(x, y, tol=tol)
%timeit downgrid_equidist(x, y, n_grid_new)
%timeit downgrid_curvature(x, y, n_grid_new)
%timeit downgrid_optimize(x, y, n_grid_new)

# Plot downgridding outputs
fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x_equi, y_equi, color="red", marker="o")
ax.plot(x_curv, y_curv, color="green", marker="o")
ax.plot(x_optim, y_optim, color="blue", marker="o")
ax.plot(x_maxtol, y_maxtol, color="magenta", marker="o")

# Check if `downgrid_maxtol(x, y, tol=0)` removes points on lines
x = np.arange(10)
y = np.array([0, 1, 2, 3, 2, 1, 2, 1, 0, -10])

downgrid_maxtol(x, y, tol=0)


#%% Optimality exploration of implemented version
# Setup
import numpy as np
import matplotlib.pyplot as plt
from randomvars import downgrid_maxtol

n = 1000
np.random.seed(101)
x = np.linspace(0, 2*np.pi, n)
y = np.sin(x) + np.random.uniform(size = n)

def plot_downgrid(x, y, tol):
    x_new, y_new = downgrid_maxtol(x, y, tol)
    plt.plot(x, y, "k")
    plt.plot(x_new, y_new, "r-o")
    # plt.show()

def max_dist(x_base, y_base, x_ref, y_ref):
    f_ref = lambda x: np.interp(x, x_ref, y_ref)
    return np.max(np.abs(f_ref(x_base) - y_base))

def downgrid_n(x, y, tol):
    x_new, y_new = downgrid_maxtol(x, y, tol)
    print(
        f"Tolerance: {tol}; Max. distance: {max_dist(x, y, x_new, y_new)}; "
        f"n_grid: {len(x_new)}."
    )
    return len(x_new)

def seq_cosine(x, y):
    x_start, y_start = x[0], y[0]
    x_end, y_end = x[-1], y[-1]
    inds = slice(1, -1)
    x_vec_start, y_vec_start = x[inds] - x_start, y[inds] - y_start
    x_vec_end,   y_vec_end   = x_end - x[inds], y_end - y[inds]
    in_product = x_vec_start * x_vec_end + y_vec_start * y_vec_end
    vec_start_len = np.sqrt(x_vec_start * x_vec_start + y_vec_start * y_vec_start)
    vec_end_len = np.sqrt(x_vec_end * x_vec_end + y_vec_end * y_vec_end)

    return in_product / (vec_start_len * vec_end_len)

# Change in grid length with respect to tolerance
tol_grid = np.linspace(0, 2, 1001)
n_grid = np.array([downgrid_n(x, y, tol) for tol in tol_grid])

plt.plot(tol_grid, n_grid)
plt.show()
## CONCLUSION: grid length **doesn't strictly decrease when tolerance
## increases**. This seems to be due to non-optimality of implemented greedy
## algorithm of `downgrid_maxtol()`.

# Show non-optimality of `downgrid_maxtol()`
cosines = seq_cosine(x, y)
max_cosine_ind = cosines.argmax()+1
x_line, y_line = x[[0, max_cosine_ind, -1]], y[[0, max_cosine_ind, -1]]

plot_downgrid(x, y, 1.948)
plt.plot(x_line, y_line, "-o")
plt.show()

x_maxtol, y_maxtol = downgrid_maxtol(x, y, 1.948)
max_dist(x, y, x_line, y_line)
max_dist(x, y, x_maxtol, y_maxtol)

x_maxtol_ind = (x == x_maxtol[1]).nonzero()[0][0]
print(f"`downgrid_maxtol()` middle point index: {x_maxtol_ind}")
print(f"'Almost straight line' middle point index: {max_cosine_ind}")
## CONCLUSION: here there is a "more optimal" grid consisting from 3 points
## which connects start and end points with almost traight line. However,
## `downgrid_maxtol()` doesn't have it as output because of its greedy
## nature: it keeps "stretching current segment" until next candidate point
## makes output have maximum distance higher then input tolerance.
## Here are findings that are aligned with these conclusions:
## - Maximum distance from "almost straight line" 3-point downgrid is less than
##   one from `downgrid_maxtol()` output 3-point downgrid.
## - Middle point of "almost straight line" downgrid is more to the left than
## one from `downgrid_maxtol()` output.
