import numpy as np
from scipy.optimize import root_scalar
import scipy.stats.distributions as distrs

from randomvars import Disc


def divvec_find(p, n_points):
    """Find optimal division vector with desired number of points

    Given probability vector `p` (with possible zeros) and desired number of
    division points `n_points` (not less than `len(p)`), find division vector
    which describes the optimal division with `n_points` points.

    Division vector `divvec` describes procedure of transforming probabilities
    `p` into "divided probabilities": `p / divvec` values each repeated
    `divvec` times (**note** that elements of division vector are at least 1).
    Number of points in division vector is a sum of its elements (which is the
    number of elements after transformation). Division vector with fixed number
    of points is said to be **optimal** if it has the smallest maximum of
    "divided probabilities" among all division vectors with the same number of
    points. **Note** that optimal division vector is often not unique.

    Parameters
    ----------
    p : numpy array
        There shouldn't be negative values.
    n_points : int
        Desired number of points in output division vector.

    Returns
    -------
    divvec : integer numpy array
        Optimal division vector with `n_points` points.
    """
    if n_points <= len(p):
        return np.ones(len(p), dtype="int64")

    # Find initial guess of optimal division vector by searching for the
    # maximum "divided probability" which corresponds to the division vector
    # with `n_points` points. This substantially reduces evaluation time.
    target = lambda prob: np.sum(divvec_from_pmax(p, prob)) - n_points
    a = np.min(p[p > 0]) / (n_points + 1)
    b = np.max(p)
    pmax_target = root_scalar(target, bracket=(a, b)).root

    divvec = divvec_from_pmax(p, pmax_target)

    # Adjusting may be needed because of possible equal elements of `p` which
    # introduce discontinuity in target function
    return divvec_adjust(p=p, n_points=n_points, divvec=divvec)


def divvec_adjust(p, n_points, divvec):
    divvec = divvec.astype("int64")

    divvec_points = np.sum(divvec)
    if divvec_points == n_points:
        return divvec

    # Remove extra division points (not executed if there is no extra points).
    # As there will be expected division by zero (resulting into infinity in
    # case of positive numerator or invalid value in case of zero numerator),
    # ignore numpy's division by zero and invalid value `RuntimeWarning`s.
    with np.errstate(divide="ignore", invalid="ignore"):
        while divvec_points > n_points:
            # Division point should be removed in such way that it tries
            # minimize predicted maximum "divided probability".
            # Note that if element of `divvec` is 1, it results into `inf` or
            # `nan` predicted "divided probability" and thus will never be
            # removed (i.e. picked as an index of the minimum value) because
            # `nanargmin()` is used instead of `argmin()`.
            pdiv_pred = p / (divvec - 1)
            divvec[np.nanargmin(pdiv_pred)] -= 1
            divvec_points -= 1
            pass

    # Add missing division points (not executed if there is no missing points)
    while divvec_points < n_points:
        # Add division point to element in such a way that it tries to decrease
        # current maximum "divided probability"
        pdiv_cur = p / divvec
        divvec[np.argmax(pdiv_cur)] += 1
        divvec_points += 1

    return divvec


def divvec_from_pmax(p, pmax):
    # Make output values to be at least 1 to overcome zero elements of `p` (for
    # which output should be exactly 1)
    return np.maximum(np.ceil(p / pmax), 1)


p = np.array([0.1, 0.4, 0.05, 0.01, 0.04, 0.25, 0.15, 0, 0])
n_points = len(p) + 10
divvec = divvec_find(p, n_points)
print(f"{divvec=}")

# Interesting behavior for i=6->i=7 and i=16->i=17 which is a result of
# non-uniqueness of optimal division vectors.
p = np.array([0, 0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3])
for i in range(1, 31):
    n_points = len(p) + i
    divvec = divvec_find(p, n_points)
    pdiv_max = np.max(p / divvec)
    print(f"{i=}: {divvec=}; {pdiv_max=}")

# p = np.random.uniform(size=2)
p = np.random.uniform(size=100)
p = p / p.sum()
n_points = 100 * len(p)
divvec = divvec_find(p, n_points)
print(f"{divvec=}")
# %timeit divvec_find(p, n_points)


# %% Explore naive upgridding using conversion
import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt
import randomvars._utilsgrid as utilsgrid


def upgrid_xp_manual(x, p, x_new):
    y = utilsgrid._y_from_xp(x, p)
    density = interp1d(x, y)
    y_new = density(x_new)
    p_new = utilsgrid._p_from_xy(x_new, y_new)
    return x_new, p_new


def upgrid_xp_naive(x, p, n_inner_points):
    x_new = augment_grid(x, n_inner_points)
    return upgrid_xp_manual(x, p, x_new)


def insert_x(x, x_to_add):
    return np.sort(np.concatenate((x, x_to_add)))


def augment_grid(x, n_inner_points):
    x_new = [
        np.linspace(x[i], x[i + 1], n_inner_points + 1, endpoint=False)
        for i in np.arange(len(x) - 1)
    ]
    x_new.append([x[-1]])
    return np.concatenate(x_new)


def xp_to_cdf(x, p):
    cump = np.cumsum(p)
    return interp1d(x, cump, kind="previous", bounds_error=False, fill_value=(0, 1))


def xy_to_cdf(x, y):
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


x = np.array([0.03032502, 0.26397697, 0.42297539, 0.78702054])
p = np.array([0.30726725, 0.07676467, 0.18088069, 0.43508739])
y = utilsgrid._y_from_xp(x, p)
interval_p = 0.5 * np.diff(x) * (y[:-1] + y[1:])

plt.plot(x, y, "-o")
plt.show()

x_grid = np.linspace(x[0], x[-1], 1001)
xp_cdf = xp_to_cdf(x, p)
xy_cdf = xy_to_cdf(x, y)
plt.plot(x_grid, xp_cdf(x_grid))
plt.plot(x_grid, xy_cdf(x_grid))
plt.show()

x_up, p_up = upgrid_xp_naive(x, p, 1)
xp_cdf = xp_to_cdf(x, p)
xp_up_cdf = xp_to_cdf(x_up, p_up)
plt.plot(x_grid, xp_cdf(x_grid), label="input")
plt.plot(x_grid, xp_up_cdf(x_grid), label="upgrid")
plt.legend()
plt.show()

# Manual upgridding
x_grid = np.linspace(x[0], x[-1], 1001)
for x_to_add in np.linspace(x[2], x[3], 11)[1:-1]:
    x_new = insert_x(x, [x_to_add])

    x_up, p_up = upgrid_xp_manual(x, p, x_new)

    # plt.plot(x, p, "o", label="input")
    # plt.plot(x_up, p_up, "o", label="upgrid")
    # plt.legend()
    # plt.show()

    xp_cdf = xp_to_cdf(x, p)
    xp_up_cdf = xp_to_cdf(x_up, p_up)
    plt.plot(x_grid, xp_cdf(x_grid), label="input")
    plt.plot(x_grid, xp_up_cdf(x_grid), label="upgrid")
    plt.legend()
    plt.show()
