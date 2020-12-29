"""
This file contains experimental versions of downgridding xp-grids.
"""
from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
import scipy.stats as ss
import scipy.stats.distributions as distrs

from randomvars import Disc
from randomvars._continuous import _xy_from_cdf_spline
import randomvars._utils as utils
import randomvars._utilsgrid as utilsgrid


# %% Downgrid xp-grid iteratively with penalty
def downgrid_xp_penalty(x, p, n_grid, metric="L2"):
    # Here `c` has all values of CDF: 0, inner cumulative probabilities, 1.
    c = np.concatenate(([0], np.cumsum(p)))

    for i in range(len(x) - n_grid):
        penalty = compute_xc_penalty(x, c, metric)

        # Pick best index as the one which delivers the smallest penalty
        min_ind = np.argmin(penalty)

        x, c = delete_xc_index(x, c, min_ind, metric=metric)

    return x, np.diff(c)


def compute_xc_penalty(x, c, metric):
    dc = np.abs(np.diff(c))
    dx = np.diff(x)

    if metric == "L2":
        inner_penalty = dx[:-1] * dx[1:] * dc[1:-1] / (dx[:-1] + dx[1:])
    if metric == "L1":
        inner_penalty = dc[1:-1] * np.minimum(dx[:-1], dx[1:])

    res = np.concatenate(([dc[0] * dx[0]], inner_penalty, [dc[-1] * dx[-1]]))

    return np.sqrt(2) * res if metric == "L2" else res


def delete_xc_index(x, c, ind, metric):
    if ind == 0:
        c = np.delete(c, 1)
    elif ind == (len(x) - 1):
        c = np.delete(c, ind)
    else:
        if metric == "L2":
            alpha = (x[ind] - x[ind - 1]) / (x[ind + 1] - x[ind - 1])
        elif metric == "L1":
            mid = 0.5 * (x[ind - 1] + x[ind + 1])
            # alpha = 1 if x < mid; alpha = 0.5 if x = mid; alpha = 0 if x > mid
            alpha = 0.5 * (0.0 + (mid < x[ind]) + (mid <= x[ind]))

        # Avoid modifing original array
        c_right = c[ind + 1]
        c = np.delete(c, ind + 1)
        c[ind] = alpha * c[ind] + (1 - alpha) * c_right

    x = np.delete(x, ind)

    return x, c


# %% Downgrid xp-grid with converting to xy-grid
def downgrid_xp_fromxy(x, p, n_grid, metric="L2"):
    y = utilsgrid._y_from_xp(x, p, metric)
    x_down, y_down = downgrid_xy(x, y, n_grid)
    p_down = utilsgrid._p_from_xy(x_down, y_down, metric)

    return x_down, p_down


# %% Downgrid xy-grid with spline method
def downgrid_xy(x, y, n_grid):
    if n_grid >= len(x):
        return x, y

    cdf_vals = utils._trapez_integral_cum(x, y)
    cdf_spline = UnivariateSpline(x=x, y=cdf_vals, s=np.inf, k=2)

    # Iteratively reduce smoothing factor to achieve xy-grid with enough points
    # Note: technically, condition should be checking number of knots in
    # derivative of `cdf_spline`, but this is the same as number of knots in
    # spline itself.
    cur_s = cdf_spline.get_residual()
    while len(cdf_spline.get_knots()) < n_grid:
        cur_s *= 0.01
        # Ensure that there is no infinite loop
        if cur_s <= 1e-16:
            break
        cdf_spline.set_smoothing_factor(cur_s)

    x_res, y_res = _xy_from_cdf_spline(cdf_spline)

    if len(x_res) < n_grid:
        # If output xy-grid has not enough points (which can happen if `n_grid`
        # is sufficiently large, use all input grid and later iteratively
        # remove points
        x_res, y_res = x, y

    return downgrid_penalty(x_res, y_res, n_grid)


def downgrid_penalty(x, y, n_grid):
    for _ in range(len(x) - n_grid):
        x, y = remove_point_from_xy(x, y)

    return x, y


def compute_penalty(x, y):
    # Compute current neighboring square of every x-grid: square of left
    # trapezoid (if no, then zero) plus square of right trapezoid (if no, then
    # zero).
    trapezoids_ext = np.concatenate(([0], 0.5 * np.diff(x) * (y[:-1] + y[1:]), [0]))
    square_cur = trapezoids_ext[:-1] + trapezoids_ext[1:]

    # Compute new neighboring square after removing corresponding x-value and
    # replacing two segments with one segment connecting neighboring xy-points.
    # The leftmost and rightmost x-values are removed without any segment
    # replacement.
    square_new = np.concatenate(([0], 0.5 * (x[2:] - x[:-2]) * (y[:-2] + y[2:]), [0]))

    # Compute penalty as value of absolute change in total square if
    # corresponding x-point will be removed.
    return np.abs(square_cur - square_new)


def remove_point_from_xy(x, y):
    penalty = compute_penalty(x, y)

    # Pick best index (including edges) as the one which delivers the smallest
    # change in total square compared to the current iteration
    min_ind = np.argmin(penalty)

    x = np.delete(x, min_ind)
    y = np.delete(y, min_ind)

    return x, y / utils._trapez_integral(x, y)


# %% Downgrid by removing elements with lowest probability
def downgrid_xp_direct(x, p, n_grid):
    sorter = np.argsort(p)
    out_inds = sorter[-n_grid:]
    x_down = x[out_inds]
    p_down = p[out_inds]
    p_down = p_down / np.sum(p_down)

    return utils._sort_parallel(x_down, y=p_down, y_name="p", warn=False)


# %% Helpers
def fun_distance_xp(xp1, xp2, metric="L2"):
    x1, p1 = xp1
    x2, p2 = xp2

    if metric == "L1":
        return ss.wasserstein_distance(
            u_values=x1, v_values=x2, u_weights=p1, v_weights=p2
        )
    elif metric == "L2":
        return ss.energy_distance(
            u_values=x1, v_values=x2, u_weights=p1, v_weights=p2
        ) / np.sqrt(2)


def xp_to_cdf(x, p):
    cump = np.cumsum(p)
    return interp1d(x, cump, kind="previous", bounds_error=False, fill_value=(0, 1))


def xp_mean(xp):
    return np.sum(xp[0] * xp[1])


# %% Experiments
rv = Disc.from_rv(distrs.binom(n=100, p=0.5))
# rv = Disc.from_rv(distrs.poisson(mu=100))
# rv = Disc.from_rv(distrs.hypergeom(200, 70, 120))
# rv = Disc.from_rv(distrs.randint(low=0, high=10))
# rv = Disc.from_rv(distrs.dlaplace(a=1))

x, p = rv.x, rv.p
xp = x, p
n_grid = 4

## Notes: "L1" downgridding usually results into less support width than "L2"
xp_down_penalty_l1 = downgrid_xp_penalty(x, p, n_grid, metric="L1")
xp_down_penalty_l2 = downgrid_xp_penalty(x, p, n_grid, metric="L2")
xp_down_fromxy = downgrid_xp_fromxy(x, p, n_grid)
xp_down_direct = downgrid_xp_direct(x, p, n_grid)

## In some cases downgridding with "L1" metric results into closer (based on
## "L2" metric) CDF than with "L2" itself. Like, for example with uniform
## discrete distribution.
print(f"{fun_distance_xp(xp, xp_down_penalty_l1, metric='L2')=}")
print(f"{fun_distance_xp(xp, xp_down_penalty_l2, metric='L2')=}")
print(f"{fun_distance_xp(xp, xp_down_fromxy)=}")
print(f"{fun_distance_xp(xp, xp_down_direct)=}")

## Note: expected value is preserved only if no edge is removed. Edge removal
## breaks expected value but usually not by much (as edge removal should be
## justified by low penalty)
print(f"{xp_mean(xp)=}")
print(f"{xp_mean(xp_down_penalty_l1)=}")
print(f"{xp_mean(xp_down_penalty_l2)=}")
print(f"{xp_mean(xp_down_fromxy)=}")
print(f"{xp_mean(xp_down_direct)=}")


# %% Timings
x = np.sort(np.random.uniform(size=1000))
p = np.random.uniform(size=1000)
p = p / p.sum()

# %timeit downgrid_xp_penalty(x, p, 10, metric="L1")
# %timeit downgrid_xp_penalty(x, p, 10, metric="L2")
# %timeit downgrid_xp_fromxy(x, p, 10)
# %timeit downgrid_xp_direct(x, p, 10)
