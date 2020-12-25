"""
Code which creates benchmarking data for different values of hyperparameters of
currently best downgridding approaches.
"""
import csv
import timeit
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import scipy.stats.distributions as distrs

from randomvars import Cont
from randomvars._continuous import _xy_from_cdf_spline
import randomvars.options as op
import randomvars._utils as utils

# Use bigger value for `Cont.from_rv()` option in order to get wider variaty
# xy-grid lengths
op.set_option("n_grid", 5001)


# %% Downgrid functions
def downgrid_spline(x, y, n_grid, params=None):
    if n_grid >= len(x):
        return x, y
    if params is None:
        params = {}

    s_small = params.get("s_small", 1e-16)
    scale_factor = params.get("scale_factor", 0.5)

    cdf_vals = utils._trapez_integral_cum(x, y)
    cdf_spline = UnivariateSpline(x=x, y=cdf_vals, s=np.inf, k=2)

    # Iteratively reduce smoothing factor to achieve xy-grid with enough points
    # Note: technically, condition should be checking number of knots in
    # derivative of `cdf_spline`, but this is the same as number of knots in
    # spline itself.
    while len(cdf_spline.get_knots()) < n_grid:
        cur_s = cdf_spline.get_residual() * scale_factor
        # Ensure that there is no infinite loop
        if cur_s <= s_small:
            break
        cdf_spline.set_smoothing_factor(cur_s)

    x_res, y_res = _xy_from_cdf_spline(cdf_spline)

    if len(x_res) < n_grid:
        # If output xy-grid has not enough points (which can happen if `n_grid`
        # is sufficiently large, use all input grid and later iteratively
        # remove points
        x_res, y_res = x, y

    return downgrid_penalty(x_res, y_res, n_grid, params=params)


def downgrid_penalty(x, y, n_grid, params=None):
    removable_edges = params.get("removable_edges", False)

    for _ in range(len(x) - n_grid):
        x, y = remove_point_from_xy(x, y, removable_edges)

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


def remove_point_from_xy(x, y, removable_edges):
    penalty = compute_penalty(x, y)

    # Pick best index (including or excluding edges) as the one which delivers
    # the smallest change in total square compared to the current iteration
    if removable_edges:
        min_ind = np.argmin(penalty)
    else:
        min_ind = np.argmin(penalty[1:-1]) + 1

    x = np.delete(x, min_ind)
    y = np.delete(y, min_ind)

    # Renormalization
    integr = utils._trapez_integral(x, y)
    if integr > 0:
        y = y / integr
    else:
        y = np.repeat(1 / (x[-1] - x[0]), len(x))

    return x, y


# %% Setup benchmarking
def bench(params, downgrid_name, n_timeit=10):
    with op.option_context({"cdf_tolerance": params["cdf_tolerance"]}):
        rv = Cont.from_rv(DISTRIBUTIONS[params["distr_name"]])

    x, y = rv.x, rv.y
    n_grid = int(max(np.floor(len(x) * params["n_grid_frac"]), 2))

    downgrid_fun = DOWNGRID_FUNCTIONS[downgrid_name]
    xy_down = downgrid_fun(x, y, n_grid, params)
    acc_down = fun_distance((x, y), xy_down)
    time_down = (
        timeit.timeit(
            lambda: downgrid_fun(x, y, n_grid, params=params), number=n_timeit
        )
        / n_timeit
    )

    return {
        **params,
        "downgrid_name": downgrid_name,
        "n_grid_input": len(x),
        "n_grid": n_grid,
        "accuracy": f"{acc_down:.2E}",
        "time_ms": np.round(1000 * time_down, decimals=5),
    }


def fun_distance(xy1, xy2):
    f = xy_to_cdf_spline(*xy1)
    g = xy_to_cdf_spline(*xy2)
    x = np.concatenate((xy1[0], xy2[0]))
    a, b = np.min(x), np.max(x)

    # Use non-zero `full_output` to suppress warnings
    return np.sqrt(
        quad(lambda x: (f(x) - g(x)) ** 2, a=a, b=b, limit=100, full_output=1)[0]
    )


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

    return cdf_spline


DISTRIBUTIONS_COMMON = {
    "beta": distrs.beta(a=10, b=20),
    "chi_sq": distrs.chi2(df=10),
    "expon": distrs.expon(),
    "f": distrs.f(dfn=20, dfd=20),
    "gamma": distrs.gamma(a=10),
    "laplace": distrs.laplace(),
    "lognorm": distrs.lognorm(s=0.5),
    "norm": distrs.norm(),
    "norm2": distrs.norm(loc=10),
    "norm3": distrs.norm(scale=0.1),
    "norm4": distrs.norm(scale=10),
    "norm5": distrs.norm(loc=10, scale=0.1),
    "t": distrs.t(df=10),
    "uniform": distrs.uniform(),
    "uniform2": distrs.uniform(loc=10, scale=0.1),
    "weibull_max": distrs.weibull_max(c=2),
    "weibull_min": distrs.weibull_min(c=2),
}

DISTRIBUTIONS_INF_DENSITY = {
    "inf_beta_both": distrs.beta(a=0.4, b=0.6),
    "inf_beta_left": distrs.beta(a=0.5, b=2),
    "inf_beta_right": distrs.beta(a=2, b=0.5),
    "inf_chi_sq": distrs.chi2(df=1),
    "inf_weibull_max": distrs.weibull_max(c=0.5),
    "inf_weibull_min": distrs.weibull_min(c=0.5),
}

DISTRIBUTIONS_HEAVY_TAILS = {
    "heavy_cauchy": distrs.cauchy(),
    "heavy_lognorm": distrs.lognorm(s=1),
    "heavy_t": distrs.t(df=2),
}

DISTRIBUTIONS = {
    **DISTRIBUTIONS_COMMON,
    **DISTRIBUTIONS_HEAVY_TAILS,
    **DISTRIBUTIONS_INF_DENSITY,
}

DOWNGRID_FUNCTIONS = {"spline": downgrid_spline, "penalty": downgrid_penalty}


scale_factor_vals = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.9]
cdf_tolerance_vals = [10 ** (-i) for i in range(1, 9)]
n_grid_frac_vals = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
removable_edges_vals = [False, True]
param_names = [
    "distr_name",
    "scale_factor",
    "cdf_tolerance",
    "n_grid_frac",
    "removable_edges",
]

params_list = [
    dict(zip(param_names, par))
    for par in product(
        DISTRIBUTIONS.keys(),
        scale_factor_vals,
        cdf_tolerance_vals,
        n_grid_frac_vals,
        removable_edges_vals,
    )
]

downgrid_names_list = DOWNGRID_FUNCTIONS.keys()


# %% Becnhmark
print(f"There are total of {len(params_list) * len(downgrid_names_list)} benchmarks.")

out_file = Path(__file__).parent.joinpath("downgrid_cont_benchmarks_output.csv")

with open(out_file, mode="w") as f:
    fields = [
        "distr_name",
        "scale_factor",
        "cdf_tolerance",
        "n_grid_frac",
        "removable_edges",
        "downgrid_name",
        "n_grid_input",
        "n_grid",
        "accuracy",
        "time_ms",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)

    writer.writeheader()

    for i, (par, down_name) in enumerate(
        product(params_list, downgrid_names_list), start=1
    ):
        print(f"#{i}: params={par}, downgrid_name={down_name}")
        bench_out = bench(params=par, downgrid_name=down_name)
        writer.writerow(bench_out)