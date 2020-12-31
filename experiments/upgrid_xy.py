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
