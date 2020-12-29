"""
This file contains experimental versions of downgridding xy-grids.
Conclusions:
- **Downgrid xy-grid**. It seems that usage of `UnivariateSpline` (combined
  with custom greedy downgridding based on probability penalty logic to ensure
  exact number of elements in output xy-grid) delivers best compromise between
  computation time and accuracy.  This is implemented in `downgrid_spline()`.
  However, some benchmarks are done in 'downgrid_cont_benchmarks.py'.
- **Downgrid xp-grid**. Use iterative greedy removing of points based on
  penalty. Implemented in `downgrid_xp` (probably should be simplified to not
  use `downgrid_metricpenalty()`), as part of other approaches to
  xy-downgridding.
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.linalg import solve_banded
from scipy.optimize import root_scalar
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm

from randomvars import Cont
import randomvars.options as op

# %% Downgrid xy-grid
# Version 1. Downgrid by iteratively removing points. At one iteration point is
# picked so that its removing will lead to **the smallest absolute change in
# square** compared to current (at the beginning of iteration) xy-grid.
def downgrid_probpenalty(x, y, n_grid_out, plot_step=10):
    x_orig, y_orig = x, y

    cur_net_change = 0

    # Iteratively remove one-by-one x-values with the smallest penalty
    for i in range(len(x) - n_grid_out):
        penalty = compute_penalty(x, y)

        # Pick best index as the one which delivers the smallest change in
        # total square compared to the current iteration
        min_ind = np.argmin(penalty)

        if (i + 1) % plot_step == 0:
            plt.plot(x, penalty, "best index")
            plt.title(f"Length of x = {len(x)}")
            plt.show()

            plt.plot(x_orig, y_orig, label="original")
            plt.plot(x, y, "-o", label="current")
            plt.plot(x[min_ind], y[min_ind], "o", label="best index")
            plt.title(f"Density. Length of x = {len(x)}")
            plt.legend()
            plt.show()

        # print(f"Delete x={x[min_ind]}")
        x = np.delete(x, min_ind)
        y = np.delete(y, min_ind)
        y = y / trapez_integral(x, y)

    return x, y / trapez_integral(x, y)


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


# Version 1.5. Downgrid by iteratively removing points. At one iteration point
# is picked so that its removing will lead to **the best balancing of removed
# square**.
def downgrid_probpenalty_1half(x, y, n_grid_out, plot_step=10):
    x_orig, y_orig = x, y

    cur_net_change = 0

    # Iteratively remove one-by-one x-values with the smallest penalty
    for i in range(len(x) - n_grid_out):
        penalty = compute_penalty_1half(x, y)

        # Pick best index as the one which best balances current net change
        # from the input.
        best_ind = np.argmin(np.abs(cur_net_change + penalty))
        cur_net_change = cur_net_change + penalty[best_ind]

        if (i + 1) % plot_step == 0:
            plt.plot(x, penalty, "best index")
            plt.title(f"Length of x = {len(x)}")
            plt.show()

            plt.plot(x_orig, y_orig, label="original")
            plt.plot(x, y, "-o", label="current")
            plt.plot(x[best_ind], y[best_ind], "o", label="best index")
            plt.title(f"Density. Length of x = {len(x)}")
            plt.legend()
            plt.show()

        # print(f"Delete x={x[best_ind]}")
        x = np.delete(x, best_ind)
        y = np.delete(y, best_ind)
        y = y / trapez_integral(x, y)

    return x, y / trapez_integral(x, y)


def compute_penalty_1half(x, y):
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

    # Compute penalty as value for which total square will change if
    # corresponding x-point will be removed.
    return square_new - square_cur


# Version 2. Downgrid by iteratively removing points. At one iteration point is
# picked so that its removing after renormalization results into **the smallest
# absolute difference between input reference CDF and new CDF at removed
# point**.
def downgrid_probpenalty_2(x, y, n_grid_out, plot_step=10):
    x_orig, y_orig = x, y
    cump_ref = trapez_integral_cum(x, y)

    # Iteratively remove one-by-one x-values with the smallest penalty
    for i in range(len(x_orig) - n_grid_out):
        penalty = compute_penalty_2(x, y, cump_ref)
        min_ind = np.argmin(penalty)
        # print(f"Delete x={x[min_ind]}")

        if (i + 1) % plot_step == 0:
            plt.plot(x, penalty, "-o")
            plt.plot(x[min_ind], penalty[min_ind], "or")
            plt.title(f"Penalty. Length of x = {len(x)}")
            plt.show()

            plt.plot(x_orig, y_orig, label="original")
            plt.plot(x, y, "-o", label="current")
            plt.plot(x[min_ind], y[min_ind], "o", label="min. penalty")
            plt.title(f"Density. Length of x = {len(x)}")
            plt.legend()
            plt.show()

        x, y, cump_ref = delete_index(x, y, cump_ref, min_ind)

    return x, y / trapez_integral(x, y)


def compute_penalty_2(x, y, cump_ref):
    integral_cum = trapez_integral_cum(x, y)
    sq_total = integral_cum[-1]
    sq_inter = np.diff(trapez_integral_cum(x, y))

    # Nearest two-squares of inner x-points (sum of squares of two nearest
    # intervals)
    sq_twointer_before = sq_inter[:-1] + sq_inter[1:]

    # Squares after removing x-point (for inner x-points only)
    sq_twointer_after = 0.5 * (y[:-2] + y[2:]) * (x[2:] - x[:-2])

    # Coefficient of stretching
    alpha = sq_total / (sq_total + (sq_twointer_after - sq_twointer_before))

    # Compute penalty as difference between input reference cump and cump after
    # removing corresponding x-points
    dx = np.diff(x)[:-1]
    dx2 = x[2:] - x[:-2]
    dy2 = y[2:] - y[:-2]
    penalty_inner = np.abs(
        cump_ref[1:-1]
        - alpha * (0.5 * dy2 * dx ** 2 / dx2 + y[:-2] * dx + cump_ref[:-2])
    )

    return np.concatenate(([cump_ref[1]], penalty_inner, [1 - cump_ref[-2]]))


# Version 3 (**VERY SLOW**). Downgrid by iteratively removing points. At one
# iteration point is picked so that its removing after renormalization results
# into the smallest functional distance between input refernce CDF and new CDF.
def downgrid_probpenalty_3(x, y, n_grid_out, plot_step=10, method="L2"):
    x_orig, y_orig = x, y
    cdf_spline_ref = xy_to_cdf_spline(x, y)

    # Iteratively remove one-by-one x-values with the smallest penalty
    for i in range(len(x_orig) - n_grid_out):
        penalty = compute_penalty_3(x, y, cdf_spline_ref, method=method)
        min_ind = np.argmin(penalty)
        print(f"Delete x={x[min_ind]}")

        if (i + 1) % plot_step == 0:
            plt.plot(x, penalty, "-o")
            plt.plot(x[min_ind], penalty[min_ind], "or")
            plt.title(f"Penalty. Length of x = {len(x)}")
            plt.show()

            plt.plot(x_orig, y_orig, label="original")
            plt.plot(x, y, "-o", label="current")
            plt.plot(x[min_ind], y[min_ind], "o", label="min. penalty")
            plt.title(f"Density. Length of x = {len(x)}")
            plt.legend()
            plt.show()

        x = np.delete(x, min_ind)
        y = np.delete(y, min_ind)
        y = y / trapez_integral(x, y)

    return x, y


def compute_penalty_3(x, y, cdf_spline_ref, method="L2"):
    sq_total = trapez_integral(x, y)

    res = []

    for i in range(len(x)):
        x_cur = np.delete(x, i)
        y_cur = np.delete(y, i)
        y_cur = y_cur * sq_total / trapez_integral(x_cur, y_cur)

        cdf_spline = xy_to_cdf_spline(x_cur, y_cur)

        res.append(fun_distance(cdf_spline, cdf_spline_ref, method=method))

    return np.array(res)


# Version from spline
def downgrid_spline(x, y, n_grid_out, s_big=2, s_small=1e-16):
    cdf_vals = trapez_integral_cum(x, y)
    cdf_spline = UnivariateSpline(x=x, y=cdf_vals, s=np.inf, k=2)

    cur_s = s_big
    scale_factor = 0.5
    while cur_s > s_small:
        n_knots = len(cdf_spline.get_knots())
        if n_knots >= n_grid_out:
            break
        cdf_spline.set_smoothing_factor(cur_s)
        cur_s *= scale_factor

    x_res, y_res = cdf_spline_to_xy(cdf_spline)
    n_excess_points = len(x_res) - n_grid_out

    if n_excess_points < 0:
        # If output xy-grid has not enough points, use all input grid and later
        # possibly remove points
        x_res, y_res = x, y
        n_excess_points = len(x_res) - n_grid_out

    if n_excess_points > 0:
        # Remove excess points if `s` got too small
        for _ in range(n_excess_points):
            x_res, y_res = remove_point_from_xy(x_res, y_res)

    return x_res, y_res


def remove_point_from_xy(x, y):
    # This uses "version 1" removing approach
    penalty = compute_penalty(x, y)

    # Pick best index (accept edges) as the one which delivers the smallest
    # change in total square compared to the current iteration
    min_ind = np.argmin(penalty[1:-1]) + 1

    x = np.delete(x, min_ind)
    y = np.delete(y, min_ind)
    y = y / trapez_integral(x, y)

    return x, y


def cdf_spline_to_xy(spline):
    dens_spline = spline.derivative()
    x = dens_spline.get_knots()
    y = np.clip(dens_spline(x), 0, None)
    y = y / trapez_integral(x, y)

    return x, y


# Downgrid xp-grid
def downgrid_xp(x, p, n_grid_out, metric="L2", plot_step=10, remove_edges=True):
    c = np.concatenate(([0], np.cumsum(p)))
    x_down, c_down = downgrid_metricpenalty(
        x=x,
        c=c,
        n_grid_out=n_grid_out,
        metric=metric,
        plot_step=plot_step,
        remove_edges=remove_edges,
    )
    p_down = np.diff(c_down)
    return x_down, p_down


def downgrid_metricpenalty(
    x, c, n_grid_out, metric="L2", plot_step=10, remove_edges=True
):
    """
    Here `c` - constant values on intervals (-inf, x[0]), [x[0], x[1]), ...,
    [x[-2], x[-1]), and [x[-1], +inf) between `x[:-1]` and `x[1:]` (this also
    means that `len(c) == len(x) + 1`)
    """
    x_orig, c_orig = x, c
    x_grid = np.linspace(x[0], x[-1], 1001)

    for i in range(len(x) - n_grid_out):
        penalty = compute_metricpenalty(x, c, metric)

        # Pick best index as the one which delivers the smallest penalty
        if remove_edges:
            min_ind = np.argmin(penalty)
        else:
            min_ind = np.argmin(penalty[1:-1]) + 1

        if (i + 1) % plot_step == 0:
            plt.plot(x, penalty, "-o")
            plt.plot(x[min_ind], penalty[min_ind], "or")
            plt.title(f"Penalty. Length of x = {len(x)}")
            plt.show()

            plt.plot(
                x_grid,
                interp1d(x_orig, c_orig[1:], kind="previous")(x_grid),
                label="original",
            )
            plt.plot(
                x_grid,
                interp1d(
                    x,
                    c[1:],
                    kind="previous",
                    bounds_error=False,
                    fill_value=(c[0], c[-1]),
                )(x_grid),
                label="current",
            )
            plt.plot(x[min_ind], c[min_ind], "o", label="best index")
            plt.title(f"Piecewise constant. Length of x = {len(x)}")
            plt.legend()
            plt.show()

        # print(f"Delete x={x[min_ind]}")
        x, c = delete_xc_index(x, c, min_ind, metric=metric)

    return x, c


def compute_metricpenalty(x, c, metric):
    dc_abs = np.abs(np.diff(c))
    dx = np.diff(x)

    if metric == "L2":
        inner_penalty = dx[:-1] * dx[1:] * dc_abs[1:-1] / (dx[:-1] + dx[1:])
    if metric == "L1":
        inner_penalty = dc_abs[1:-1] * np.minimum(dx[:-1], dx[1:])

    res = np.concatenate(([dc_abs[0] * dx[0]], inner_penalty, [dc_abs[-1] * dx[-1]]))

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


# Version from xp. Retype xy-grid to xp-grid, downgrid xp-grid, retype back to
# xy-grid.
def downgrid_fromxp(x, y, n_grid_out, plot_step=10, metric="L2"):
    p = p_from_xy(x, y)
    x_down, p_down = downgrid_xp(
        x,
        p,
        n_grid_out=n_grid_out,
        metric=metric,
        plot_step=plot_step,
        remove_edges=False,
    )
    # Clip possible negative values
    y_down = np.clip(y_from_xp(x_down, p_down), 0, None)
    y_down = y_down / trapez_integral(x_down, y_down)

    return x_down, y_down


def y_from_xp(x, p, metric="L2"):
    metric_coeffs = {"L1": 0.75, "L2": 2 / 3, "Linf": 0.5}
    coeff = metric_coeffs[metric]
    dx = np.diff(x)
    dx_lead = np.concatenate([dx, [0]])
    dx_lag = np.concatenate([[0], dx])

    banded_matrix = 0.5 * np.array(
        [dx_lag * (1 - coeff), (dx_lag + dx_lead) * coeff, dx_lead * (1 - coeff)]
    )

    return solve_banded(l_and_u=(1, 1), ab=banded_matrix, b=p)


def p_from_xy(x, y, metric="L2"):
    metric_coeffs = {"L1": 0.75, "L2": 2 / 3, "Linf": 0.5}
    coeff = metric_coeffs[metric]
    cump = trapez_integral_cum(x, y)
    dx = np.diff(x)

    # This is missing last value, which is 1
    disc_cump = cump[:-1] + 0.5 * dx * (coeff * y[:-1] + (1 - coeff) * y[1:])

    p = np.diff(disc_cump, prepend=0, append=1)

    return p


# Version from slopes. Convert xy-grid to xslope-grid, downgrid with discrete
# downgridding, convert back to xy-grid.
def downgrid_fromslopes(x, y, n_grid_out, metric="L2", plot_step=10):
    slopes = np.diff(y) / np.diff(x)
    c = np.concatenate(([0], slopes, [0]))
    x_down, c_down = downgrid_metricpenalty(
        x, c, n_grid_out=n_grid_out, metric=metric, plot_step=plot_step
    )

    y0 = np.interp(x_down[0], x, y)
    y_down = np.concatenate(([y0], y0 + np.cumsum(c_down[1:-1] * np.diff(x_down))))
    y_down = np.clip(y_down, 0, None)
    y_down = y_down / trapez_integral(x_down, y_down)

    return x_down, y_down


# Helper functions
def trapez_integral_cum(x, y):
    """Compute cumulative integral with trapezoidal formula.
    Element of output represents cumulative probability **before** its left "x"
    edge.
    """
    res = np.cumsum(0.5 * np.diff(x) * (y[:-1] + y[1:]))
    return np.concatenate([[0], res])


def trapez_integral(x, y):
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def delete_index(x, y, cump_ref, ind):
    x = np.delete(x, ind)
    y = np.delete(y, ind)
    y = y / trapez_integral(x, y)
    cump_ref = np.delete(cump_ref, ind)
    return x, y, cump_ref


def xy_to_cdf_spline(x, y):
    dens_spline = UnivariateSpline(x=x, y=y, k=1, s=0)
    cdf_spline_raw = dens_spline.antiderivative()

    def cdf_spline(t):
        t = np.atleast_1d(t)
        res = np.zeros(len(t))
        t_is_in = (x[0] < t) & (t <= x[-1])
        res[t_is_in] = cdf_spline_raw(t[t_is_in])
        res[x[-1] < t] = 1
        return res

    cdf_spline.get_knots = lambda: cdf_spline_raw.get_knots()

    return cdf_spline


def fun_distance(f, g, method="L2"):
    knots = np.concatenate([f.get_knots(), g.get_knots()])
    a, b = np.min(knots), np.max(knots)

    if method == "L2":
        return np.sqrt(quad(lambda x: (f(x) - g(x)) ** 2, a=a, b=b)[0])
    elif method == "L1":
        return quad(lambda x: np.abs(f(x) - g(x)), a=a, b=b)[0]


# %% Experiments
# with op.option_context({"cdf_tolerance": 1e-6}):
#     rv = Cont.from_rv(ss.norm())
# rv = Cont.from_rv(ss.expon())
rv = Cont.from_rv(ss.beta(a=3, b=5))
# rv = Cont.from_rv(ss.laplace())
x, y = rv.x, rv.y

rv_cdf = xy_to_cdf_spline(x, y)

# n_grid_out = int(np.floor(np.sqrt(len(x))))
# n_grid_out = len(x) - 30
n_grid_out = 3

x_down_pen, y_down_pen = downgrid_probpenalty(x, y, n_grid_out, plot_step=10000)
rv_down_pen_cdf = xy_to_cdf_spline(x_down_pen, y_down_pen)

x_down_pen_1half, y_down_pen_1half = downgrid_probpenalty_1half(
    x, y, n_grid_out, plot_step=10000
)
rv_down_pen_1half_cdf = xy_to_cdf_spline(x_down_pen_1half, y_down_pen_1half)

# x_down_pen2, y_down_pen2 = downgrid_probpenalty_2(x, y, n_grid_out, plot_step=1000)
# rv_down_pen2_cdf = xy_to_cdf_spline(x_down_pen2, y_down_pen2)

# x_down_pen3, y_down_pen3 = downgrid_probpenalty_3(x, y, n_grid_out, plot_step=1000)
# rv_down_pen3_cdf = xy_to_cdf_spline(x_down_pen3, y_down_pen3)

x_down_spline, y_down_spline = downgrid_spline(x, y, n_grid_out)
rv_down_spline_cdf = xy_to_cdf_spline(x_down_spline, y_down_spline)

x_down_fromxp, y_down_fromxp = downgrid_fromxp(x, y, n_grid_out, plot_step=10000)
rv_down_fromxp_cdf = xy_to_cdf_spline(x_down_fromxp, y_down_fromxp)

x_down_fromslopes, y_down_fromslopes = downgrid_fromslopes(
    x, y, n_grid_out, plot_step=10000
)
rv_down_fromslopes_cdf = xy_to_cdf_spline(x_down_fromslopes, y_down_fromslopes)

print(f"Target number of points: {n_grid_out}")
print(f"{len(x_down_pen)=}")
print(f"{len(x_down_pen_1half)=}")
print(f"{len(x_down_spline)=}")
print(f"{len(x_down_fromxp)=}")
print(f"{len(x_down_fromslopes)=}")

# print(f"{fun_distance(rv_cdf, rv_down_equi_cdf)=}")
print(f"{fun_distance(rv_cdf, rv_down_pen_cdf)=}")
print(f"{fun_distance(rv_cdf, rv_down_pen_1half_cdf)=}")
# print(f"{fun_distance(rv_cdf, rv_down_pen2_cdf)=}")
# print(f"{fun_distance(rv_cdf, rv_down_pen3_cdf)=}")
print(f"{fun_distance(rv_cdf, rv_down_spline_cdf)=}")
print(f"{fun_distance(rv_cdf, rv_down_fromxp_cdf)=}")
print(f"{fun_distance(rv_cdf, rv_down_fromslopes_cdf)=}")

plt.plot(x, y, "-o", label="input")
# plt.plot(x_down_equi, y_down_equi, "-o", label="equidistant")
plt.plot(x_down_pen, y_down_pen, "-o", label="probpenalty")
plt.plot(x_down_pen_1half, y_down_pen_1half, "-o", label="probpenalty_1half")
# plt.plot(x_down_pen2, y_down_pen2, "-o", label="probpenalty2")
# plt.plot(x_down_pen3, y_down_pen3, "-o", label="probpenalty3")
plt.plot(x_down_spline, y_down_spline, "-o", label="spline")
plt.plot(x_down_fromxp, y_down_fromxp, "-o", label="fromxp")
plt.plot(x_down_fromslopes, y_down_fromslopes, "-o", label="fromslopes")
plt.legend()
plt.show()

x_ext = np.linspace(x[0], x[-1], 10001)
plt.plot(x_ext, rv_cdf(x_ext), "-", label="input")
# plt.plot(x_ext, rv_down_equi_cdf(x_ext), "-", label="equidistant")
plt.plot(x_ext, rv_down_pen_cdf(x_ext), "-", label="probpenalty")
plt.plot(x_ext, rv_down_pen_1half_cdf(x_ext), "-", label="probpenalty_1half")
# plt.plot(x_ext, rv_down_pen2_cdf(x_ext), "-", label="probpenalty2")
# plt.plot(x_ext, rv_down_pen3_cdf(x_ext), "-", label="probpenalty3")
plt.plot(x_ext, rv_down_spline_cdf(x_ext), "-", label="spline")
plt.plot(x_ext, rv_down_fromxp_cdf(x_ext), "-", label="fromxp")
plt.plot(x_ext, rv_down_fromslopes_cdf(x_ext), "-", label="fromslopes")
plt.legend()
plt.show()


# %% Downgrid 3-gridder to 2-gridder
def downgrid_threegridder(x, y):
    """Both `x` and `y` should have three elements"""
    cdf_spline = xy_to_cdf_spline(x, y)
    a = x[0]
    b = x[-1]

    moment_1 = quad(lambda t: cdf_spline(t) * (t - a), a=a, b=b, limit=100)[0]
    moment_2 = quad(lambda t: cdf_spline(t) * (t - a) ** 2, a=a, b=b, limit=100)[0]

    dx = b - a
    y0 = -12 * (5 * moment_2 - 4 * dx * moment_1) / (dx ** 4)
    # slope = 40 * (4 * moment_2 - 3 * dx * moment_1) / (dx ** 5)
    # y2 = slope * dx + y0
    y2 = y[0] * (x[1] - x[0]) / dx + y[1] + y[2] * (x[2] - x[1]) / dx - y0

    x_res = np.array([a, b])
    y_res = np.array([y0, y2])

    return x_res, y_res


def downgrid_threegridder_2(x, y):
    """Both `x` and `y` should have three elements"""
    cdf_spline = xy_to_cdf_spline(x, y)
    a = x[0]
    b = x[-1]

    dx = b - a
    moment = quad(lambda t: cdf_spline(t) * (t - a) * (b - t), a=a, b=b, limit=100)[0]
    y_mean = (cdf_spline(x[-1]) - cdf_spline(x[0])) / dx
    # Convert from array to float
    y_mean = y_mean[0]

    y0 = 30 * moment / dx ** 4 - 1.5 * y_mean
    y2 = 2 * y_mean - y0

    x_res = np.array([a, b])
    y_res = np.array([y0, y2])

    return x_res, y_res


def downgrid_threegridder_3(x, y, metric="L2"):
    """Downgridding is done by averaging two slopes"""
    slopes = (y[:-1] - y[1:]) / (x[:-1] - x[1:])

    if metric == "L2":
        alpha = (x[1] - x[0]) / (x[2] - x[0])
    elif metric == "L1":
        mid = 0.5 * (x[2] - x[0])
        # alpha = 1 if x < mid; alpha = 0.5 if x = mid; alpha = 0 if x > mid
        alpha = 0.5 * (0.0 + (mid < x[1]) + (mid <= x[1]))

    slope_res = alpha * slopes[0] + (1 - alpha) * slopes[1]

    square_cur = 0.5 * np.sum((x[1:] - x[:-1]) * (y[:-1] + y[1:]))
    dx = x[-1] - x[0]
    y0 = square_cur / dx - 0.5 * slope_res * dx
    y1 = slope_res * dx + y0

    return x[[0, -1]], np.array([y0, y1])


def quadratic_fit(f, a, b):
    """
    Output function of the form `g(x) = alpha*(x - a)**2 + beta*(x-a) + gamma`
    is best L2 fit to `f` with "fixed ends" (f(a) = g(a), f(b) = g(b)).
    """
    gamma = f(a)[0]
    g_deriv = lambda t: (t - a) * (t - b)
    g_deriv_sqnorm = quad(lambda t: g_deriv(t) ** 2, a=a, b=b)[0]
    f_integral = quad(lambda t: f(t) * g_deriv(t), a=a, b=b)[0]

    f_coeff = ((f(b) - f(a)) / (b - a))[0]
    leftover_integral = quad(lambda t: f_coeff * (t - a) * g_deriv(t), a=a, b=b)[0]

    alpha = (f_integral - leftover_integral) / g_deriv_sqnorm
    beta = f_coeff - alpha * (b - a)

    return lambda t: alpha * (t - a) ** 2 + beta * (t - a) + gamma


x = np.array([0, 1, 3])
y = np.array([2, 1, 1.25])
# y = y / trapez_integral(x, y)
x_down_2, y_down_2 = downgrid_threegridder_2(x, y)
x_down_3, y_down_3 = downgrid_threegridder_3(x, y, metric="L2")

cdf_ref = xy_to_cdf_spline(x, y)
cdf_down_2 = xy_to_cdf_spline(x_down_2, y_down_2)
cdf_down_3 = xy_to_cdf_spline(x_down_3, y_down_3)

fun_distance(cdf_ref, cdf_down_2)
fun_distance(cdf_ref, cdf_down_3)

x_plot = np.linspace(x[0], x[-1], 1001)
plt.plot(x, y, label="input")
plt.plot(x_down_2, y_down_2, label="downgrid_2")
plt.plot(x_down_3, y_down_3, label="downgrid_3")
plt.legend()
plt.show()

plt.plot(x_plot, cdf_ref(x_plot), label="input")
plt.plot(x_plot, cdf_down_2(x_plot), label="downgridded")
plt.legend()
plt.show()

f = xy_to_cdf_spline(x, y)
a = x[0]
b = x[-1]
g = quadratic_fit(f, a, b)

tmp = UnivariateSpline(x=x_plot, y=f(x_plot), k=2, s=100)

f(a), f(b)
g(a), g(b)

x_plot = np.linspace(a, b, 10001)
plt.plot(x_plot, f(x_plot), label="input")
plt.plot(x_plot, g(x_plot), label="fit")
plt.plot(x_plot, tmp(x_plot), label="spline")
plt.legend()
plt.show()


tmp_deriv = UnivariateSpline(x=x_plot, y=np.gradient(f(x_plot), x_plot), k=1, s=1000)
plt.plot(x_plot, np.gradient(f(x_plot), x_plot), label="f derivative")
plt.plot(x_plot, np.gradient(g(x_plot), x_plot), label="g derivative")
plt.plot(x_plot, tmp_deriv(x_plot), label="spline")
plt.legend()
plt.show()


# %% Downgrid 4-gridder to 3-gridder
def downgrid_fourgridder(x, y, ksi):
    alpha, beta = ksi - x[0], x[-1] - ksi
    z1, z2 = 0, np.trapz(y, x)
    y_ksi = (2 * (z2 - z1) - (alpha * y[0] + beta * y[-1])) / (x[-1] - x[0])
    return np.array([x[0], ksi, x[-1]]), np.array([y[0], y_ksi, y[-1]])


def fun_distance_xy(xy1, xy2, method="L2"):
    cdf_spline1 = xy_to_cdf_spline(xy1[0], xy1[1])
    cdf_spline2 = xy_to_cdf_spline(xy2[0], xy2[1])
    return fun_distance(cdf_spline1, cdf_spline2, method=method)


x = np.array([0, 1, 2, 10])
y = np.array([0, 10, 5, 5])
y = y / np.trapz(y, x)

ksi_arr = np.linspace(x[0], x[-1], 101)[1:-1]
xy_down_ksi = [downgrid_fourgridder(x, y, ksi) for ksi in ksi_arr]

ksi_penalty = np.array([fun_distance_xy((x, y), xy_down) for xy_down in xy_down_ksi])
best_ksi_ind = np.argmin(ksi_penalty)

## Plot
cmap = cm.plasma
norm = matplotlib.colors.Normalize(vmin=ksi_penalty.min(), vmax=ksi_penalty.max())

fig, ax = plt.subplots()

for (x_down, y_down), penalty in zip(xy_down_ksi, ksi_penalty):
    ax.plot(x_down, y_down, "-o", color=cmap(norm(penalty)))
ax.plot(x, y, "k", label="input")
ax.plot(*xy_down_ksi[best_ksi_ind], color="red")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm)

plt.show()
