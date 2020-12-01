import numpy as np
from scipy.linalg import solve_banded
from randomvars import Cont, Disc
import randomvars._utils as utils


# %% Functions
def y_from_xp(x, p, coeff):
    dx = np.diff(x)
    dx_lead = np.concatenate([dx, [0]])
    dx_lag = np.concatenate([[0], dx])

    banded_matrix = 0.5 * np.array(
        [dx_lag * (1 - coeff), (dx_lag + dx_lead) * coeff, dx_lead * (1 - coeff)]
    )

    return solve_banded(l_and_u=(1, 1), ab=banded_matrix, b=p)


def cont_from_disc(rv, method="mean"):
    x, p = rv.x, rv.p
    if method == "mean":
        y = y_from_xp(x, p, coeff=2 / 3)
    elif method == "median":
        y = y_from_xp(x, p, coeff=0.75)
    elif method == "midpoint":
        y1 = y_from_xp(x, p, coeff=0.5 - 1e-8)
        y2 = y_from_xp(x, p, coeff=0.5 + 1e-8)

        y = 0.5 * (y1 + y2)

    return Cont(x=x, y=y)


def p_from_xy(x, y, coeff):
    cum_p = utils._trapez_integral_cum(x, y)
    dx = np.diff(x)

    # This is missing last value, which is 1
    disc_cum_p = cum_p[:-1] + 0.5 * dx * (coeff * y[:-1] + (1 - coeff) * y[1:])

    p = np.diff(disc_cum_p, prepend=0, append=1)

    return p


def disc_from_cont(rv, method):
    method_coeffs = {"mean": 2 / 3, "median": 0.75, "midpoint": 0.5}
    coeff = method_coeffs[method]

    x, y = rv.x, rv.y

    p = p_from_xy(x, y, coeff)

    return Disc(x=x, p=p)


def disc_from_rv(rv, x_new, method):
    """
    Notes:
    - This approach results into discrete random variable that is closest to
    `rv` in terms of some distance (for "median" `method` - Wasserstein, for
    "mean" - Cramer) among discrete random variables with values (even zero)
    located at `x_new`.
    - If `x_new` is wider than `rv`'s support, then using `method` "mean"
    preserves expected value of random variable. This is true thanks to formula
    `E[X] = integral_{0}^{+infinity}{(1 - F_X(x))dx}` (if X can take only
    positive values, but preservation of mean is true for any `X` as long as
    `x_new` is wider than its support). Expected value of what is preserved in
    case `x_new` is not wider than `rv`'s support remains to be an open
    question.
    """
    # This is missing last value, which is 1
    disc_cum_p = compute_interval_centers(f=rv.cdf, grid=x_new, method=method)

    p = np.diff(disc_cum_p, prepend=0, append=1)

    return Disc(x=x_new, p=p)


def cont_from_rv(rv, x_new, method):
    """
    Notes:
    - If `x_new` is inside `rv`'s support, behavior at edges is similar to
    winsorizing: tail probabilities get "squashed" into first/last interval of
    piecewise-continuous density.
    """
    # First convert to discrete
    disc = disc_from_rv(rv=rv, x_new=x_new, method=method)

    # Then convert discrete to Cont
    method_coeffs = {"mean": 2 / 3, "median": 0.75}
    coeff = method_coeffs[method]

    x, p = disc.x, disc.p
    y = y_from_xp(x, p, coeff)

    # Seems like this correction is needed but it seems "hacky"
    y = np.maximum(0, y)

    return Cont(x=x, y=y)


def cont_from_rv_derivative(rv, x_new):
    """
    Notes:
    - This approach seems to be 'bad' because it can't reproduce initial `rv`
    in case of a 'good' `x_new`.
    - However, if `x_new` is inside `rv`'s support, then behaviour at edges is
    similar to trimming: tails are just removed.
    """
    cdf_vals = rv.cdf(x_new)
    y_vals = np.gradient(cdf_vals, x_new)

    return Cont(x=x_new, y=y_vals)


def compute_interval_centers(f, grid, method):
    if method == "median":
        # This assumes `f` is monotonic
        grid_centers = 0.5 * (grid[:-1] + grid[1:])
        return f(grid_centers)
    elif method == "mean":
        grid_diff = grid[1:] - grid[:-1]
        integrals = np.array(
            [utils._quad_silent(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1)]
        )
        return integrals / grid_diff


# %% Experiments
# x = np.array([0, 1, 2])
# y = np.array([0, 1, 0])

# x = np.array([0, 1, 2])
# y = np.repeat(1, len(x)) / np.max(x)

from scipy.stats import norm

my_norm = Cont.from_rv(norm())
x, y = my_norm.x, my_norm.y

coeff = 0.5 + 1e-8
# coeff = 0.000001

# Both uniform and triangular distributions result into [0.25, 0.5, 0.25] probabilities
# when `coeff = 0.5`. This is a reason why transform from xp-grid to xy-grid is
# not correctly defined in that case.
# Using `0.5 + eps` solves this problem while introducing approximation errors.
p = p_from_xy(x, y, coeff=coeff)
y_new = y_from_xp(x, p, coeff=coeff)

np.max(np.abs(y - y_new))


# %% Experiments with general approach
import matplotlib.pyplot as plt
import scipy.stats.distributions as distrs

# rv = distrs.expon()
# rv = Cont(x=[0, 1, 2], y=[0, 1, 0])
rv = Cont.from_rv(distrs.norm())

# x_new = np.array([-10, -3, -0.5, 0.5, 3, 10])
# x_new = rv.ppf(q=np.linspace(0, 1, 9))
# x_new = rv.ppf(q=[0.1, 0.9])
# x_new = np.sort(np.random.uniform(rv.x[0], rv.x[-1], size=11))
x_new = rv.x

# Convert to Disc
my_rv_disc_mean = disc_from_rv(rv=rv, x_new=x_new, method="mean")
my_rv_disc_median = disc_from_rv(rv=rv, x_new=x_new, method="median")
plt.plot(my_rv_disc_mean.x, my_rv_disc_mean.p, "-o", label="mean")
plt.plot(my_rv_disc_median.x, my_rv_disc_median.p, "-o", label="median")
plt.legend()
plt.show()

# Convert to Cont
my_rv_cont_mean = cont_from_rv(rv=rv, x_new=x_new, method="mean")
my_rv_cont_median = cont_from_rv(rv=rv, x_new=x_new, method="median")
my_rv_cont_derivative = cont_from_rv_derivative(rv=rv, x_new=x_new)
plt.plot(x_new, rv.pdf(x_new), "-ok", label="original")
plt.plot(my_rv_cont_mean.x, my_rv_cont_mean.y, "-o", label="mean")
plt.plot(my_rv_cont_median.x, my_rv_cont_median.y, "-o", label="median")
plt.plot(my_rv_cont_derivative.x, my_rv_cont_derivative.y, "-o", label="derivative")
plt.legend()
plt.show()

my_rv_cont_derivative_disc = disc_from_rv(my_rv_cont_derivative, x_new, "mean")
plt.plot(my_rv_disc_mean.x, my_rv_disc_mean.p, label="mean")
plt.plot(my_rv_disc_median.x, my_rv_disc_median.p, label="median")
plt.plot(my_rv_cont_derivative_disc.x, my_rv_cont_derivative_disc.p, label="derivative")
plt.legend()
plt.show()
