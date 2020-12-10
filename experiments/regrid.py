import numpy as np
import scipy.stats as ss
from scipy.integrate import quad


def approx_cdf_piececont(
    f, f_inv, x_condition, cump_condition, method="mean", max_steps=100, thres=1e-8
):
    x_is_free = np.isnan(x_condition)
    x_fixed_val = x_condition[~x_is_free]
    cump_is_free = np.isnan(cump_condition)
    cump_fixed_val = cump_condition[~cump_is_free]

    center_fun = construct_center_fun(f, method)

    cump_prev = init_cump(cump_condition)
    x_prev = x_from_cump(cump_prev, f_inv, x_is_free, x_fixed_val)
    steps = [(x_prev, cump_prev)]

    for i in range(1, max_steps):
        cump_cur = cump_from_x(x_prev, center_fun, cump_is_free, cump_fixed_val)
        x_cur = x_from_cump(cump_cur, f_inv, x_is_free, x_fixed_val)
        steps.append((x_cur, cump_cur))

        if stop_rule(
            step_cur=(x_cur, cump_cur), step_prev=(x_prev, cump_prev), thres=thres
        ):
            break

        x_prev = x_cur
        cump_prev = cump_cur

    return steps


def init_cump(cump_condition):
    """Initialize p-vector
    Every non-conditioned value is initialized to be proportional probability
    apart from nearest supplied conditions (after adding 0 and 1 to edges).
    """
    cump_ext = np.concatenate([[0], cump_condition, [1]])
    x_equi = np.arange(len(cump_ext))

    inds = np.isfinite(cump_ext)

    return np.interp(x_equi[1:-1], x_equi[inds], cump_ext[inds])


def x_from_cump(p, f_inv, x_is_free, x_fixed_val):
    res = np.zeros(len(p) + 1)

    cump_ext = np.concatenate([[0], p, [1]])
    cump_mid = 0.5 * (cump_ext[:-1] + cump_ext[1:])

    res[x_is_free] = f_inv(cump_mid[x_is_free])
    res[~x_is_free] = x_fixed_val

    return res


def cump_from_x(x, center_fun, cump_is_free, cump_fixed_val):
    res = np.zeros(len(cump_is_free))

    x_left = x[np.concatenate([cump_is_free, [False]])]
    x_right = x[np.concatenate([[False], cump_is_free])]

    res[cump_is_free] = center_fun(x_left, x_right)
    res[~cump_is_free] = cump_fixed_val

    return res


def construct_center_fun(f, method):
    if method == "mean":
        # Use faster integration method if available
        if "integrate" in dir(f):
            integr_f = f.integrate
        else:
            integr_f = lambda a, b: quad(f, a, b)[0]

        def center_fun(a, b):
            res = np.zeros(len(a))

            a_less = a < b

            a1 = a[a_less]
            b1 = b[a_less]
            res[a_less] = np.array([integr_f(l, r) for l, r in zip(a1, b1)]) / (b1 - a1)

            res[~a_less] = f(a[~a_less])

            return res

    elif method == "median":

        def center_fun(a, b):
            return f(0.5 * (a + b))

    return center_fun


def stop_rule(step_cur, step_prev, thres):
    x_error = relative_error(step_cur[0], step_prev[0])
    cump_error = relative_error(step_cur[1], step_prev[1])

    print(f"{x_error=:10.9f}; {cump_error=:10.9f}")

    return (x_error < thres) and (cump_error < thres)


def relative_error(x, y):
    return np.sum(np.abs(x - y)) / np.sum(np.abs(y))


def p_from_cump(cump):
    return np.diff(cump, prepend=0, append=1)


# rv = ss.beta(a=3, b=3)
rv = ss.norm()
f = rv.cdf
f_inv = rv.ppf

cump_condition = np.repeat(np.nan, 100)
x_condition = np.repeat(np.nan, 101)
# cump_condition = np.array([0.1, np.nan, 0.75, 0.95, 0.99])
# x_condition = np.concatenate([[-4], np.repeat(np.nan, len(cump_condition) - 1), [4]])

steps = approx_cdf_piececont(f, f_inv, x_condition, cump_condition, method="mean")

x, cump = steps[-1]

# np.array([quad(f, x_l, x_r)[0] / (x_r - x_l) for x_l, x_r in zip(x[:-1], x[1:])])
np.array([f(0.5 * (x_l + x_r)) for x_l, x_r in zip(x[:-1], x[1:])])
cump
