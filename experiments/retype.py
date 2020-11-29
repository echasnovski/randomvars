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


def disc_to_cont(rv, method="mean"):
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


# %% Experiments
# x = np.array([0, 1, 2])
# y = np.array([0, 1, 0])

from scipy.stats import norm

my_norm = Cont.from_rv(norm())
x, y = my_norm.x, my_norm.y
coeff = 0.5

p = p_from_xy(x, y, coeff=coeff)
y_new = y_from_xp(x, p, coeff=coeff + 1e-8)

np.max(np.abs(y - y_new))
