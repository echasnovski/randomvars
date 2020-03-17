import numpy as np
from scipy.stats.distributions import rv_continuous, norm, beta, cauchy, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#%% Functions
def trapez_integral(x, y):
    """ Compute integral with trapezoidal formula.
    """
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def is_local_extremum(y):
    y = np.asarray(y)

    if len(y) == 1:
        return np.array([True])

    # Note that if `len(y)` is 2 these arrays are empty (which they should)
    left = y[:-2]
    inn = y[1:-1]
    right = y[2:]

    inn_loc_min = ((left >= inn) & (inn <= right)) & ((left > inn) | (inn < right))
    inn_loc_max = ((left <= inn) & (inn >= right)) & ((left < inn) | (inn > right))

    return np.concatenate(
        (
            # First element is local extremum if first segment isn't horizontal
            [y[0] != y[1]],
            inn_loc_min | inn_loc_max,
            # Last element is local extremum if last segment isn't horizontal
            [y[-2] != y[-1]],
        )
    )


class PieceLin(rv_continuous):
    def __init__(self, x, y, *args, **kwargs):
        integral = np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))

        self._x = np.asarray(x)
        self._y = np.asarray(y) / integral

        # Set support
        kwargs["a"] = self.a = self._x[0]
        kwargs["b"] = self.b = self._x[-1]

        super(PieceLin, self).__init__(*args, **kwargs)

    def _pdf(self, x, *args):
        return np.interp(x, self._x, self._y)


def abs_curvature(x, y):
    deriv = np.gradient(y, x)
    deriv2 = np.gradient(deriv, x)
    return np.abs(deriv2) / (1 + deriv ** 2) ** 1.5


def piecelin_quantiles(x, y, probs):
    rv = PieceLin(x, y)
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


def regrid_quantile(x, y, n_grid):
    rv = PieceLin(x, y)
    x_new = rv.ppf(np.linspace(0, 1, n_grid))
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
n_grid_new = 11
x_equi, y_equi = regrid_equidist(x, y, n_grid_new)
x_curv, y_curv = regrid_curvature(x, y, n_grid_new)
x_quant, y_quant = regrid_quantile(x, y, n_grid_new)
x_optim, y_optim = regrid_optimize(x, y, n_grid_new)

dist_grid((x, y), (x_equi, y_equi), method="maxabs")
dist_grid((x, y), (x_curv, y_curv), method="maxabs")
dist_grid((x, y), (x_quant, y_quant), method="maxabs")
dist_grid((x, y), (x_optim, y_optim), method="maxabs")

dist_grid((x, y), (x_equi, y_equi), method="meanabs")
dist_grid((x, y), (x_curv, y_curv), method="meanabs")
dist_grid((x, y), (x_quant, y_quant), method="meanabs")
dist_grid((x, y), (x_optim, y_optim), method="meanabs")

trapez_integral(x_equi, y_equi)
trapez_integral(x_curv, y_curv)
trapez_integral(x_quant, y_quant)
trapez_integral(x_optim, y_optim)

# Plot regridding outputs
fig, ax = plt.subplots()
# ax.set_xlim([-7000, 7000])
ax.plot(x, y)
ax.plot(x_equi, y_equi, color="red", marker="o")
ax.plot(x_curv, y_curv, color="green", marker="o")
# ax.plot(x_quant, y_quant)
ax.plot(x_optim, y_optim, color="blue", marker="o")

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
