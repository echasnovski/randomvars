import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import scipy.stats as ss
from randomvars._continuous import Cont, _detect_finite_supp, _combine_grids
import randomvars.options as op

N_GRID_PLOT = 1001


def cdf_grid(rv, n_grid):
    x_left, x_right = _detect_finite_supp(rv, supp=None, small_prob=1e-6)

    # Construct equidistant grid
    x_equi = np.linspace(x_left, x_right, n_grid)

    # Construct quantile grid
    prob_left, prob_right = rv.cdf([x_left, x_right])
    prob_equi = np.linspace(prob_left, prob_right, n_grid)
    x_quan = rv.ppf(prob_equi)

    # Combine equidistant and quantile grids into one sorted array
    return _combine_grids(x_equi, x_quan)


def create_cdf_spline(rv, n_grid, **kwargs):
    x_grid = cdf_grid(rv, n_grid)
    return UnivariateSpline(x=x_grid, y=rv.cdf(x_grid), **kwargs)


def cont_from_cdf_spline(rv, n_grid, **kwargs):
    cdf_spline = create_cdf_spline(rv, n_grid, **kwargs)
    x_grid_spline = cdf_spline.get_knots()
    y_grid_spline = cdf_spline.derivative(n=1)(x_grid_spline)
    return Cont(x=x_grid_spline, y=np.clip(y_grid_spline, a_min=0, a_max=None))


def plot_spline(spline, **kwargs):
    knots = spline.get_knots()
    x_grid = np.linspace(knots[0], knots[-1], N_GRID_PLOT)
    plt.plot(x_grid, spline(x_grid), **kwargs)


def plot_spline_deriv(spline, **kwargs):
    knots = spline.get_knots()
    x_grid = np.linspace(knots[0], knots[-1], N_GRID_PLOT)
    plt.plot(x_grid, spline.derivative(n=1)(x_grid), **kwargs)


def plot_rv(rv, fun_name="cdf", *args, **kwargs):
    f = getattr(rv, fun_name)
    p_grid = np.linspace(1e-6, 1 - 1e-6, N_GRID_PLOT)
    x_grid = rv.ppf(p_grid)
    plt.plot(x_grid, f(x_grid), *args, **kwargs)


def fun_diff(f, g, method="L2"):
    if method == "L1":
        integrand = lambda x: np.abs(f(x) - g(x))
    elif method == "L2":
        integrand = lambda x: (f(x) - g(x)) ** 2

    res = quad(integrand, -np.inf, np.inf, limit=100)[0]
    res = np.sqrt(res) if method == "L2" else res
    return res


rv = ss.norm()
# rv = ss.gamma(a=10)

# rv = ss.beta(a=0.5, b=0.5)
# rv = ss.chi2(df=1)
# rv = ss.chi2(df=2)
# rv = ss.weibull_max(c=0.5)

# rv = ss.cauchy()

n_grid = 1001
s = 1e-8
k = 2
cdf_spline = create_cdf_spline(rv, n_grid, s=s, k=k)
x_grid_spline = cdf_spline.get_knots()
y_grid_spline = cdf_spline.derivative(n=1)(x_grid_spline)
cum_p_grid_spline = cdf_spline(x_grid_spline)

plot_spline(cdf_spline, label="spline")
plot_rv(rv, label="original")
plt.plot(x_grid_spline, cum_p_grid_spline, "ok")
plt.legend()
plt.show()

plot_spline_deriv(cdf_spline, label="spline")
plot_rv(rv, fun_name="pdf", label="original")
plt.plot(x_grid_spline, y_grid_spline, "ok")
plt.legend()
plt.show()

# Comparing with current `Cont`
with op.option_context({"n_grid": n_grid}):
    rv_cont = Cont.from_rv(rv)
rv_spline = cont_from_cdf_spline(rv, n_grid, s=s, k=k)

print(rv_cont)
print(rv_spline)

plot_rv(rv, "pdf", color="black", label="original")
plot_rv(rv_cont, "pdf", color="blue", label="Cont")
plt.plot(rv_cont.x, rv_cont.y, color="blue", marker="o")
plot_rv(rv_spline, "pdf", color="red", label="spline")
plt.plot(rv_spline.x, rv_spline.y, color="red", marker="o")
plt.legend()
plt.show()

fun_diff(rv.pdf, rv_cont.pdf)
fun_diff(rv.pdf, rv_spline.pdf)
fun_diff(rv_cont.pdf, rv_spline.pdf)

fun_diff(rv.cdf, rv_cont.cdf)
fun_diff(rv.cdf, rv_spline.cdf)
fun_diff(rv_cont.cdf, rv_spline.cdf)

# Timings
# with op.option_context({"n_grid": n_grid}):
#     %timeit Cont.from_rv(rv)

# %timeit cont_from_cdf_spline(rv, n_grid, s=s, k=k)

# CURRENT CONCLUSION: using splines to find xy-grid (by approximating CDF with
# quadratic spline) with `s=1e-8` gives:
# - Pro++: around 8 to 10 times fewer grid elements.
# - Pro: approximates CDF on par with current method.
# - Con: not so well approximates density (L2 distance is around 1e-4 against 1e-5).
# - Con: takes about 8 to 10 times more computation time (around 10 ms vs around 1 ms).
# - Cont: it doesn't scale with `n_grid` very well in terms of execution time.
